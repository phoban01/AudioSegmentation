//
//  main.cpp
//  HPCP
//
//  Created by ITMA user on 30/07/2014.
//  Copyright (c) 2014 ITMA user. All rights reserved.
//

//#if !defined(ARMA_64BIT_WORD)
//#define ARMA_64BIT_WORD
//#endif

#include <iostream>
#include <cmath>

#include "essentia/essentia.h"
#include "essentia/essentiamath.h"
#include "essentia/algorithm.h"
#include "essentia/algorithmfactory.h"
#include "essentia/streaming/algorithms/poolstorage.h"
#include "essentia/scheduler/network.h"

#include "armadillo/armadillo"
#include "mlpack/core.hpp"
#include "mlpack/methods/neighbor_search/neighbor_search.hpp"

#include <numeric>      // std::accumulate

using namespace arma;
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;
using namespace mlpack::neighbor;

Mat<double> RP(std::vector<std::vector<float>>input,int m, int v);

double mean(std::vector<double>x);
double auto_threshold(std::vector<double>data);
std::vector<double> peak_detection(std::vector<double>data);
std::vector<float> normalize(std::vector<float> input);
std::vector<std::vector<float>> normalize2D(std::vector<std::vector<float>> input);
std::vector<float> derivative(std::vector<float> input);
double cos_distance(vector<float> v1, vector<float> v2);
double e_distance(std::vector<float> v1, std::vector<float> v2);
double enorm(std::vector<float> x,std::vector<float> y);
double enorm(colvec x,colvec y);
double enormR(rowvec x,rowvec y);
mat ckernel(int n,double sigma);
std::vector<double> correlate(Mat<double> matrix,Mat<double> kernel);
std::vector<double> normalize(std::vector<double> input);
void write_audacity_labels(std::vector<double> peaks);

double cos_distance(std::vector<float> v1, std::vector<float> v2);

void write_gnu_plot_file(string path,int size);


//1. Need to perform downsample on analysis data... see serra phd thesis Qmax overview
//2. Add other descriptors that Serra uses in thesis?
//3. Thresholding...

string matrix_path = "/Users/itma/Documents/HPCP/similarity_matrix.txt";

int main(int argc, const char * argv[])
{

    string audioFilename = argv[1];

    essentia::init();
    
    Pool pool;
    
    Real sampleRate = 44100.0;
    int frameSize = 2048;
    int hopSize = frameSize * 0.5; // 1024 75% overlap @ 4096 frameSize
    float frame_duration = frameSize / sampleRate;

    AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();
    
    Algorithm* audio = factory.create("EasyLoader",
                                      "filename", audioFilename,
                                      "sampleRate", sampleRate);

    Algorithm* duration = factory.create("Duration","sampleRate",sampleRate);

    Algorithm* fc    = factory.create("FrameCutter",
                                      "frameSize", frameSize,
                                      "hopSize", hopSize);
    
    Algorithm* w     = factory.create("Windowing",
                                      "type", "hamming","size",frameSize);
    
    Algorithm* spec  = factory.create("Spectrum","size",frameSize);
    Algorithm* peaks  = factory.create("SpectralPeaks");
    Algorithm* hpcp = factory.create("HPCP");
    Algorithm* mfcc = factory.create("MFCC");

    /////////// CONNECTING THE ALGORITHMS ////////////////
    // Audio -> FrameCutter
    audio->output("audio")    >>  fc->input("signal");
    audio->output("audio")    >> duration->input("signal");

    // FrameCutter -> Windowing -> Spectrum
    fc->output("frame")       >>  w->input("frame");
    w->output("frame")        >>  spec->input("frame");
    
    spec->output("spectrum")  >>  peaks->input("spectrum");
    spec->output("spectrum")  >> mfcc->input("spectrum");
    peaks->output("frequencies")    >>  hpcp->input("frequencies");
    peaks->output("magnitudes")    >> hpcp->input("magnitudes");
    
    hpcp->output("hpcp") >> PC(pool,"HPCP");
    mfcc->output("mfcc") >> PC(pool,"MFCC");
    mfcc->output("bands") >> PC(pool,"Bands");

    duration->output("duration") >> PC(pool,"Duration");
    
    /////////// STARTING THE ALGORITHMS //////////////////
    // create a network with our algorithms...
    Network n(audio);
    // ...and run it, easy as that!
    cout << "\nLoading Audio & Performing Analysis..." << endl;
    
    n.run();
    n.clear();

    cout << "Analysis Complete" << endl;
    
    std::vector<std::vector<float>> hpcp_vector = pool.value<vector<vector<float>>>("MFCC");
    
    essentia::shutdown();
    
    int v = 25;
    int m = 10;
    int w_offset = (m - 1)/2.0;
    
    Mat<double> rp = RP(hpcp_vector,m,v);
    
    write_gnu_plot_file(matrix_path,rp.size());

    Mat<double> kernel = ckernel(atoi(argv[2]),atof(argv[3]));
    
    ofstream novelty("/Users/itma/Documents/HPCP/novelty.txt");

    std::vector<double> novelty_curve = normalize(correlate(rp, kernel));
    
    for (int i = 0 ; i < novelty_curve.size();++i) {
        novelty << i << " " << novelty_curve[i] << endl;
    }

    novelty.close();
    system("gnuplot '/Users/itma/Documents/HPCP/novelty.gnu'");

    std::vector<double> novelty_peaks = peak_detection(novelty_curve);
    for (int i = 0; i < novelty_peaks.size(); ++i) {
        novelty_peaks[i] = (novelty_peaks[i] / novelty_curve.size()) * pool.value<Real>("Duration");
    }
    cout << "Peaks at: " << novelty_peaks << endl;
    
    write_audacity_labels(novelty_peaks);
    
    return 0;
}

void write_audacity_labels(std::vector<double> peaks)
{
    ofstream audacity_labels;
    
    audacity_labels.open("/Users/itma/Documents/HPCP/audacity_labels.txt");
    
    for (int i = 0; i < peaks.size() ; ++i) {
        audacity_labels << peaks[i] << "\tEvent\n";
    }
    audacity_labels.close();
}

double e_distance(std::vector<float> v1, std::vector<float> v2)
{
    int N = (int)v1.size();
    float dot = 0.0;
    for (int n = 0; n < N; ++n)
    {
        dot += ((v1[n] - v2[n])*(v1[n] - v2[n]));
    }
    return sqrt(dot);
}

double enorm(std::vector<float> x,std::vector<float> y)
{
    double m_sum = 0.0;
    int n = (int)x.size();
    for (int i=0;i < n;++i) {
        m_sum += pow((x[i]-y[i]),2);
    }
    return sqrt(m_sum);
}

double enorm(colvec x,colvec y)
{
    double m_sum = 0.0;
    int n = x.size();
    for (int i=0;i < n;++i) {
        m_sum += pow((x(i)-y(i)),2);
    }
    return sqrt(m_sum);
}

double enormR(rowvec x,rowvec y)
{
    double m_sum = 0.0;
    int n = x.size();
    for (int i=0;i < n;++i) {
        m_sum += pow((x(i)-y(i)),2);
    }
    return sqrt(m_sum);
}

std::vector<std::vector<float>> embed(std::vector<std::vector<float>> input,int m)
{
    int t = 1;
    int n = (int)input.size() - ((m-1)*t);
    
    std::vector<std::vector<float>> output;
    std::vector<float> tmp;

    for (int i=0;i<n;++i) {
        for (int j=0;j<m;++j) {
            for (int k=0;k<input[0].size();++k)
                tmp.push_back(input[i+j][k]);
        }
        output.push_back(tmp);
        tmp.clear();
    }
    cout << "Embed Done" << endl;
    return output;
}

int wrapIndex(int x,int n)
{
    if (x < 0) {
        return n + x;
    } else if (x > 0) {
        return x % n;
    } else {
        return x;
    }
}


double gaussian(double x, double mu, double sigma) {
    return exp( -(((x-mu)/(sigma))*((x-mu)/(sigma)))/2.0 );
}

std::vector<float>  gausswin(int n,double sigma)
{
    std::vector<float> window;
    window.resize(n);
    int half_n = n / 2;
    double sum = 0;
    for (int i= -(half_n);i <= half_n;++i) {
        window[i+half_n] = gaussian(i,0,sigma);
        sum += window[i+half_n];
    }

    for (int i=0;i<n;++i) {
        window[i] /= sum;
    }
    
    return window;
}

int reflect(int M, int x)
{
    if(x < 0)
    {
        return -x - 1;
    }
    if(x >= M)
    {
        return 2*M - x - 1;
    }
    
    return x;
}

Mat<double> downsample(Mat<double> X,int v)
{
    int n = X.n_cols;
    mat H;
    for (int i=0; i < n/v;++i) {
        Mat<double> Y = X.cols(i*v,((i+1)*v)-1);
        colvec sumy = sum(Y,1);
        H.insert_cols(i,sumy / max(sumy));
    }

    return H;
}

mat vec2mat(std::vector<std::vector<float>>input)
{
    int cols = (int)(input.size());
    int rows = (int)input[0].size();
    
    mat X(rows,cols);

    for (int i = 0; i < cols ;++i) {
        for (int j=0; j < rows; ++j) {
            X(j,i) = input[i][j];
        }
    }
    
    return X;
}

std::vector<std::vector<float>> mat2vec(Mat<double> input)
{
    std::vector<std::vector<float>> out;
    for (int i = 0; i < input.n_cols;++i) {
        std::vector<float> t;
        for (int j = 0;j < input.n_rows;++j) {
            t.push_back(input(j,i));
        }
        out.push_back(t);
        t.clear();
    }
    return out;
}

int col_includes(Col<size_t> x,int j) {
    int val = 0;
    
    for (int i = 0; i < x.size(); ++i) {
        if (x(i) == j) {
            val = 1;
            break;
        }
    }
    
    return val;
}


Mat<double> RP(std::vector<std::vector<float>>input, int m, int v)
{

    
    mat reduced = vec2mat(input);
    
    reduced = downsample(reduced,v);

    Mat<double> embedded = vec2mat(embed(mat2vec(reduced),m));
//    std::vector<std::vector<float>> embedded = embed(mat2vec(reduced),m);

    int n = (int)embedded.size();
    int K = n*0.001;

    cout << "Searching for " << K << " Neighbors..." << endl;
    
    AllkNN a(embedded);
    Mat<size_t> resultingNeighbors;
    mat resultingDistances;
    a.Search(K,resultingNeighbors,resultingDistances);
    
    int n2 = resultingNeighbors.n_cols;
    //    int n2 = n;

    Mat<double> R(n2,n2);
    
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n2; ++j) {
            R(i,j) = col_includes(resultingNeighbors.col(i),j) && col_includes(resultingNeighbors.col(j),i);
        }
        cout << "-------> " << (i/(float)n2)*100 << "%...\r";
        cout.flush();
    }
    
    cout << endl;
    cout << "Matrix Written" << endl;
    
    
    Mat<double> L(n2,n2);
    int k;
    
    for (int i = 0; i < n2;++i) {
        for (int j = 0;j < n2;++j) {
//            k = wrapIndex(i + (j - 2),n2);
//            L(i,j) = R((k+1)%n2,j);
            if ((i - j) > 0) {
                L(i,j) = R(i,i-j);
            } else {
                L(i,j) = 0;
            }
        }
        cout << "-------> " << (i/(float)n2)*100 << "%...\r";
        cout.flush();
        }
    cout << endl;
//
//    int win_size = 70;
//    int halfw = win_size / 2;
//    double sigma = 20;
//    std::vector<float> coeffs = gausswin(win_size,sigma);
//    
//    Mat<float> temp(n2,n2,fill::zeros);
//    float sum,x1,y1;
//    
////    // along y - direction
//    for(int y = 0; y < n2; y++){
//        for(int x = 0; x < n2; x++){
//            sum = 0.0;
//            for(int i = halfw*-1; i <= halfw; i++){
//                y1 = reflect(n2, y - i);
//                sum = sum + coeffs[i + halfw]*L(y1, x);
//            }
//            temp(y,x) = sum;
//        }
//    }
//    
//    // along x - direction
//    for(int y = 0; y < n2; y++){
//        for(int x = 0; x < n2; x++){
//            sum = 0.0;
//            for(int i =halfw*-1; i <= halfw; i++){
//                x1 = reflect(n2, x - i);
//                sum = sum + coeffs[i + halfw]*L(y, x1);
//            }
//            L(y,x) = sum;
//        }
//    }
//
    R = L;
    
    double rmax = R.max();
    
    ofstream matrix_file;
    matrix_file.open(matrix_path);
    
    for (int i = 0; i < R.n_rows; ++i) {
        for (int j=0; j < R.n_cols; ++j) {
            matrix_file << R(i,j)  << " ";
        }
        matrix_file << "\n";
    }
    matrix_file.close();

    return R;
}

void write_gnu_plot_file(string path,int size)
{
    ofstream gnuplot_file;
    gnuplot_file.open("/Users/itma/Documents/HPCP/simx_matrix.gnu");
    
    gnuplot_file << "reset\n";
    gnuplot_file << "set term png size 1000,750\n";
    gnuplot_file << "set title 'RP'\n";
    gnuplot_file << "set output '/Users/itma/Documents/HPCP/simx_matrix.png' \n";
//    gnuplot_file << "set palette defined (0 'white', 0.5 'yellow', 1 'red')\n";
    gnuplot_file << "set palette gray\n";
    gnuplot_file << "set pm3d map\n";
    gnuplot_file << "splot '" << matrix_path << "' matrix title ''\n";
    gnuplot_file.close();
    system("gnuplot '/Users/itma/Documents/HPCP/simx_matrix.gnu'");
}

double mean(std::vector<double>x)
{
    double sum = std::accumulate(x.begin(),x.end(),0.0);
    return sum / x.size();
}

double auto_threshold(std::vector<double>data)
{
    double e = 0.001;
    double c1 = data[0];
    double c2 = data[1];
    double lastc1 = c1;
    double lastc2 = c2;
    while (true) {
        std::vector<double> class1,class2;
        for (int i = 0; i < data.size(); ++i) {
            if (abs(c1 - data[i]) < abs(c2 - data[i])) {
                class1.push_back(data[i]);
            } else {
                class2.push_back(data[i]);
            }
        }
        c2 = mean(class2);
        c1 = mean(class1);
        if ((abs(lastc2 - c2) < e) and (abs(lastc1 - c1) < e)) {
            if (class1.size() > class2.size()) {
                return c1;
            } else {
                return c2;
            }
        }
        lastc2 = c2;
        lastc1 = c1;
    }
}
std::vector<double> peak_detection(std::vector<double>data)
{
    double e = auto_threshold(data);
    std::vector<double> p,t;
    int a = 0,b = 0,d = 0;
    int i = -1;
    int xl = ((int)data.size() - 1);
    while (i != xl) {
        ++i;
        if (d == 0) {
            if (data[a] >= (data[i] + e)) {
                d = 2;
            } else if (data[i] >= (data[b] + e)) {
                d = 1;
            }
            if (data[a] <= data[i]) {
                a = i;
            } else if (data[i] <= data[b]) {
                b = i;
            }
        } else if (d == 1) {
            if (data[a] <= data[i]) {
                a = i;
            } else if (data[a] >= (data[i] + e)) {
                p.push_back(a);
                b = i;
                d = 2;
            }
        } else if (d == 2) {
            if (data[i] <= data[b]) {
                b = i;
            } else if (data[i] >= (data[b] + e)) {
                t.push_back(b);
                a = i;
                d = 1;
            }
        }
    }
    //    cout << p << endl;
    return p;
}

double cos_distance(std::vector<float> v1, std::vector<float> v2)
{
    int N = (int)v1.size();
    float dot = 0.0;
    float mag1 = 0.0;
    float mag2 = 0.0;
    int n;
    for (n = 0; n < N; ++n)
    {
        dot += v1[n] * v2[n];
        mag1 += pow(v1[n], 2);
        mag2 += pow(v2[n], 2);
    }
    return exp((dot / (sqrt(mag1) * sqrt(mag2)))-1);
}


mat ckernel(int n,double sigma)
{
    mat checker_kernel(n,n);
    
    int t;
    
    double half_n = n / 2.0;
   // double sigma = 4.25;
    double r,s = (n * 0.5) * sigma * sigma;
    double sum = 0.0;
    
    //checkboard kernel with gaussian smoothing
    for (int i = 0;  i < n;  ++i)
    {
        double x = i - half_n;
        for (int j = 0;  j < n;  ++j)
        {
            double y = j - half_n;
            r = sqrt(x*x + y*y);
            if (i<half_n) {
                if (j<half_n) {
                    t=1;
                } else {
                    t=-1;
                }
            } else {
                if (j<half_n) {
                    t=-1;
                } else {
                    t=1;
                }
            }
            checker_kernel.at(i,j) = (exp(-(r*r)/s))/(M_PI * s) * t;
            sum += checker_kernel.at(i,j);
        }
    }
    
    for (int i=0; i < n; ++i) {
        for (int j=0; j < n; ++j) {
            checker_kernel.at(i,j) /= sum;
        }
    }
    
    ofstream kernelfile("/Users/itma/Documents/HPCP/kernel.txt");
    
    for (int i=0; i < n;++i) {
        for (int j=0;j < n;++j) {
            kernelfile << i << " " << j << " " << checker_kernel(i,j) << endl;
        }
        kernelfile << endl;
    }
    kernelfile.close();
    system("gnuplot '/Users/itma/Documents/HPCP/kernel.gnu'");
    return checker_kernel;
}

std::vector<double> correlate(Mat<double> matrix,Mat<double> kernel)
{
    int n = kernel.n_rows;
    std::vector<double> novelty;

    for (int i=0; i < (sqrt(matrix.size()) - (n -1)) ;++i) {
        double corr = 0.0;
        if (i == 0) {
            for (int j=n/2 ; j < n  ; ++j) {
                for (int k=n/2 ; k < n ; ++k) {
                    corr += matrix(i+j,i+k) * kernel(j,k);
                }
            }
        } {
            for (int j=0 ; j < n  ; ++j) {
                for (int k=0 ; k < n ; ++k) {
                    corr += matrix(i+j,i+k) * kernel(j,k);
                }
            }
        }
        novelty.push_back(corr);
    }
    
    return novelty;
}

std::vector<double> normalize(std::vector<double> input)
{
    std::vector<double> output;
    double max = *max_element(input.begin(),input.end());
    double min = *min_element(input.begin(),input.end());
    
    for (int i=1;i<input.size();++i) {
        output.push_back((input[i] - min) / (max-min));
    }
    return output;
}