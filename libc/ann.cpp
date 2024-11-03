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

bool is_load = 0;

bool isIP;


using IndexNSG = ann::NSG;
using IndexHNSW = ann::HNSW<ann::FP32Quantizer<ann::Metric::IP>>;
std::unique_ptr<ann::GraphSearcherBase> searcher;

void *ann_init(int K_features, int R, const char *metric){
    ann_R = R + 80;
    ann_L = R + 20;
    std::string metricS(metric);
    if (metric == "IP") {
        isIP = true;
        void* vidx1 = ann::create_hnsw("IP", "FP32", K_features, ann_R, ann_L).release();
        return vidx1;

    } else {
        isIP = false;
        IndexNSG* vidx = new IndexNSG(K_features, metricS, ann_R, ann_L);

        vidx->nndescent_iter = nndescent_iter;
        vidx->GK = nndescent_GK;
        vidx->nndescent_S = nndescent_S;
        vidx->nndescent_R = nndescent_R;
        vidx->nndescent_L = nndescent_L;


        return (void*)vidx;
    }
}

void ann_free(void *ptr){
    if (!isIP) {
        IndexNSG* vidx = (IndexNSG*)ptr;
        delete vidx;
    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
        delete vidx1;
    }
}

void ann_add(void *ptr, int n, float * x,  const char *store){
    if (!isIP) {

        IndexNSG* vidx = (IndexNSG*)ptr;
        if (!is_load) {
            vidx->Build(x, n);
        }


        if (store != NULL) {
            vidx->final_graph.save(std::string(store));
        }


        searcher = std::move(ann::create_searcher(std::move(vidx->final_graph), vidx->metric, "SQ8U"));


        searcher->SetData(x, n, vidx->d);

    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
        if (!is_load) {
            vidx1->Build(x, n);
        }


        if (store != NULL) {
            vidx1->final_graph.save(std::string(store));
        }


        searcher = std::move(ann::create_searcher(std::move(vidx1->final_graph), "IP", "FP32"));


        searcher->SetData(x, n, vidx1->quant.dim());
    }
}

void set_ann_ef(void *ptr, int ann_ef){
    if (!isIP) {
        IndexNSG* vidx = (IndexNSG*)ptr;
    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
    }
    searcher->SetEf(ann_ef);
}


void ann_search(void *ptr, int n, const float* x, int k, float* distances,
                int32_t* labels, int num_p){
    if (!isIP) {
        IndexNSG* vidx = (IndexNSG*)ptr;
        // 调用c++函数
#pragma omp parallel for num_threads(num_p)
        for (int i = 0; i < n; ++i) {
            size_t offset = i * vidx->d;
            searcher->Search(x + offset, k, labels + i * k);
        }
    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
#pragma omp parallel for num_threads(num_p)
        for (int i = 0; i < n; ++i) {
            size_t offset = i * vidx1->quant.dim();
            searcher->Search(x + offset, k, labels + i * k);
        }

    }
}



void ann_load(void *ptr, const char *path){
    if (!isIP) {
        IndexNSG* vidx = (IndexNSG*)ptr;
        vidx->final_graph.load(std::string(path));
    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
        vidx1->final_graph.load(std::string(path));
    }
    is_load = 1;
    
}

void ann_save(void *ptr, const char *path){
    if (!isIP) {
        IndexNSG* vidx = (IndexNSG*)ptr;
        vidx->final_graph.save(std::string(path));
    } else {
        IndexHNSW* vidx1 = (IndexHNSW*)ptr;
        vidx1->final_graph.save(std::string(path));
    }
}


