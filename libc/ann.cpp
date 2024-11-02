//
#include "ann.h"
// 这里include所有c++的头文件并进行调用

#include <thread>
#include <omp.h>

#include "ann/builder.hpp"
#include "ann/nsg/nsg.hpp"
#include "ann/searcher/graph_searcher.hpp"
#include "ann/hnsw/hnsw.hpp"



int nndescent_iter = 15;
int nndescent_GK = 200;
int nndescent_S = 10;
int nndescent_R = 100;
int nndescent_L = 200;

int ann_R = 50;
int ann_M = 8;
int ann_L = 100;

bool is_load = false;

ann::Metric temp_metric;

using IndexHNSWL2 = ann::HNSW<ann::FP32Quantizer<ann::Metric::L2>>;
using IndexHNSWIP = ann::HNSW<ann::FP32Quantizer<ann::Metric::IP>>;
std::unique_ptr<ann::GraphSearcherBase> searcher;

void* ann_init(int K_features, int R, const char* metric) {
    ann_R = R;
    ann_L = R + 50;
    temp_metric = ann::metric_map[metric];

    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = new IndexHNSWL2(K_features, ann_R, ann_L);

        return (void*)vidx;
    } else {
        IndexHNSWIP* vidx = new IndexHNSWIP(K_features, ann_R, ann_L);

        return (void*)vidx;
    }
}

void ann_free(void* ptr) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;

        delete vidx;
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;

        delete vidx;
    }    
}

void ann_add(void* ptr, int n, float* x, const char* store) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;

        if (!is_load) {
            vidx->Build(x, n);
        }

        if (store != NULL) {
            vidx->final_graph.save(std::string(store));
        }

        searcher = std::move(ann::create_searcher(std::move(vidx->final_graph), "L2", "FP32"));
        searcher->SetData(x, n, vidx->quant.dim());
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;

        if (!is_load) {
            vidx->Build(x, n);
        }

        if (store != NULL) {
            vidx->final_graph.save(std::string(store));
        }

        searcher = std::move(ann::create_searcher(std::move(vidx->final_graph), "IP", "FP32"));
        searcher->SetData(x, n, vidx->quant.dim());

        delete vidx;
    }
    
}

void set_ann_ef(void* ptr, int ann_ef) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;
    }
    searcher->SetEf(ann_ef);
}

void ann_search(void* ptr, int n, const float* x, int k, float* distances,
    int32_t* labels, int num_p) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;

#pragma omp parallel for num_threads(num_p)
        for (int i = 0; i < n; ++i) {
            size_t offset = i * vidx->quant.dim();
            searcher->Search(x + offset, k, labels + i * k);
        }
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;

#pragma omp parallel for num_threads(num_p)
        for (int i = 0; i < n; ++i) {
            size_t offset = i * vidx->quant.dim();
            searcher->Search(x + offset, k, labels + i * k);
        }
    }


}

void ann_load(void* ptr, const char* path) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;

        vidx->final_graph.load(std::string(path));
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;

        vidx->final_graph.load(std::string(path));
    }
    
    is_load = true;
}

void ann_save(void* ptr, const char* path) {
    if (temp_metric == ann::Metric::L2) {
        IndexHNSWL2* vidx = (IndexHNSWL2*)ptr;

        vidx->final_graph.save(std::string(path));
    } else {
        IndexHNSWIP* vidx = (IndexHNSWIP*)ptr;

        vidx->final_graph.save(std::string(path));
    } 
}