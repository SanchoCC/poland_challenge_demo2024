//
#include "ann.h"
// 这里include所有c++的头文件并进行调用

#include <thread>
#include <omp.h>

#include "ann/builder.hpp"
#include "ann/nsg/nsg.hpp"
#include "ann/searcher/graph_searcher.hpp"



int nndescent_iter = 20;
int nndescent_GK = 170;
int nndescent_S = 10;
int nndescent_R = 100;
int nndescent_L = 200;

int ann_R = 50;
int ann_M = 8;
int ann_L = 100;

bool is_load = 0;

int original_dim = 0;
int reduced_dim = 128;

using IndexNSG = ann::NSG;
std::unique_ptr<ann::GraphSearcherBase> searcher;

void reduce_dim_PCA(const float* data, int n, int d, int target_dim, float*& reduced_data) {
    std::vector<float> mean(d, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            mean[j] += data[i * d + j];
        }
    }
    for (int j = 0; j < d; ++j) {
        mean[j] /= n;
    }

    std::vector<float> centered_data(n * d);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            centered_data[i * d + j] = data[i * d + j] - mean[j];
        }
    }

    std::vector<float> covariance(d * d, 0.0f);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += centered_data[k * d + i] * centered_data[k * d + j];
            }
            covariance[i * d + j] = sum / (n - 1);
            covariance[j * d + i] = covariance[i * d + j];
        }
    }

    std::vector<float> eigenvalues(d, 0.0f);
    std::vector<float> eigenvectors(d * d, 0.0f);
    for (int i = 0; i < d; ++i) eigenvectors[i * d + i] = 1.0f;

    for (int iter = 0; iter < 100; ++iter) {
        int p = 0, q = 1;
        float max_val = fabs(covariance[p * d + q]);
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                if (fabs(covariance[i * d + j]) > max_val) {
                    max_val = fabs(covariance[i * d + j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (max_val < 1e-6) break;

        float phi = 0.5 * atan2(2 * covariance[p * d + q], covariance[p * d + p] - covariance[q * d + q]);
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);

        for (int i = 0; i < d; ++i) {
            float temp_p = cos_phi * eigenvectors[i * d + p] - sin_phi * eigenvectors[i * d + q];
            float temp_q = sin_phi * eigenvectors[i * d + p] + cos_phi * eigenvectors[i * d + q];
            eigenvectors[i * d + p] = temp_p;
            eigenvectors[i * d + q] = temp_q;
        }
    }

    reduced_data = new float[n * target_dim];
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < target_dim; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < d; ++j) {
                sum += centered_data[i * d + j] * eigenvectors[j * d + k];
            }
            reduced_data[i * target_dim + k] = sum;
        }
    }
}

void* ann_init(int K_features, int R, const char* metric) {
    ann_R = R + 30;
    ann_L = R + 50;
    original_dim = K_features;

    std::string metricS(metric);
    IndexNSG* vidx = new IndexNSG(reduced_dim, metricS, ann_R, ann_L);

    vidx->nndescent_iter = nndescent_iter;
    vidx->GK = nndescent_GK;
    vidx->nndescent_S = nndescent_S;
    vidx->nndescent_R = nndescent_R;
    vidx->nndescent_L = nndescent_L;
    return (void*)vidx;
}

void ann_free(void *ptr){
    IndexNSG *vidx = (IndexNSG *) ptr;
    delete vidx;
}

void ann_add(void* ptr, int n, float* x, const char* store) {
    IndexNSG* vidx = (IndexNSG*)ptr;

    float* reduced_data = nullptr;
    reduce_dim_PCA(x, n, original_dim, reduced_dim, reduced_data);

    if (!is_load) {
        vidx->Build(reduced_data, n);
    }

    if (store != NULL) {
        vidx->final_graph.save(std::string(store));
    }

    searcher = std::move(ann::create_searcher(std::move(vidx->final_graph), vidx->metric, "SQ8U"));
    searcher->SetData(reduced_data, n, reduced_dim);

    delete[] reduced_data;
}

void set_ann_ef(void *ptr, int ann_ef){
    IndexNSG *vidx = (IndexNSG *)ptr;
    searcher->SetEf(ann_ef);
}


void ann_search(void *ptr, int n, const float* x, int k, float* distances,
                int32_t* labels, int num_p){
    IndexNSG *vidx = (IndexNSG *)ptr;
    // 调用c++函数
#pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < n; ++i) {
        size_t offset = i * vidx->d;
        searcher->Search(x + offset, k, labels + i * k);
    }
}



void ann_load(void *ptr, const char *path){
    IndexNSG *vidx = (IndexNSG *)ptr;
    vidx->final_graph.load(std::string(path));
    is_load = 1;
    
}

void ann_save(void *ptr, const char *path){
    IndexNSG *vidx = (IndexNSG *)ptr;
    vidx->final_graph.save(std::string(path));
}


