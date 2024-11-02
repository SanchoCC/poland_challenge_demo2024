//
#include "ann.h"
// 这里include所有c++的头文件并进行调用

#include <thread>
#include <omp.h>

#include "ann/builder.hpp"
#include "ann/nsg/nsg.hpp"
#include "ann/searcher/graph_searcher.hpp"
#include "ann/hnsw/hnsw.hpp"
#include "ann/quant/quant.hpp"



int nndescent_iter = 15;
int nndescent_GK = 200;
int nndescent_S = 10;
int nndescent_R = 100;
int nndescent_L = 200;

int ann_R = 50;
int ann_M = 8;
int ann_L = 100;

bool is_load = 0;

std::unique_ptr<ann::GraphSearcherBase> searcher;

void* ann_init(int K_features, int R, const char* metric) {
    std::string metricS(metric);
    return (void*)ann::create_hnsw(metricS, "FP16", K_features, R, R + 100).release();
}

void ann_free(void* ptr) {
    auto* hnsw = static_cast<ann::HNSW<ann::FP16Quantizer<ann::Metric::L2>>*>(ptr);
    delete hnsw;
}

void ann_add(void* ptr, int n, float* x, const char* store) {
    auto* hnsw = static_cast<ann::HNSW<ann::FP16Quantizer<ann::Metric::L2>>*>(ptr);

    if (!is_load) {
        hnsw->Build(x, n);
    }

    if (store != nullptr) {
        hnsw->final_graph.save(std::string(store));
    }

    if (hnsw->quant.metric() == ann::Metric::L2) {
        searcher = std::move(ann::create_searcher(std::move(hnsw->final_graph), "L2", "FP16"));
    } else if (hnsw->quant.metric() == ann::Metric::IP) {
        searcher = std::move(ann::create_searcher(std::move(hnsw->final_graph), "IP", "FP16"));
    }

    searcher->SetData(x, n, hnsw->quant.dim());
}

void set_ann_ef(void* ptr, int ann_ef) {
    searcher->SetEf(ann_ef);
}

void ann_search(void* ptr, int n, const float* x, int k, float* distances, int32_t* labels, int num_p) {
    auto* hnsw = static_cast<ann::HNSW<ann::FP16Quantizer<ann::Metric::L2>>*>(ptr);

#pragma omp parallel for num_threads(num_p)
    for (int i = 0; i < n; ++i) {
        size_t offset = i * hnsw->quant.dim();
        searcher->Search(x + offset, k, labels + i * k);
    }
}

void ann_load(void* ptr, const char* path) {
    auto* hnsw = static_cast<ann::HNSW<ann::FP16Quantizer<ann::Metric::L2>>*>(ptr);
    hnsw->final_graph.load(std::string(path));
    is_load = true;
}

void ann_save(void* ptr, const char* path) {
    auto* hnsw = static_cast<ann::HNSW<ann::FP16Quantizer<ann::Metric::L2>>*>(ptr);
    hnsw->final_graph.save(std::string(path));
}