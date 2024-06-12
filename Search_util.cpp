#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include "State_util.h"
//#include <iostream>  //Could not find moduleが出る

extern "C" {

void test_search_util() {
    printf("test search util\n");
}

// Treeの作成関数
Tree* createTree() {
    Tree* newTree = (Tree*)malloc(sizeof(Tree));
    if (newTree == NULL) {
        perror("Failed to allocate memory for new tree");
        exit(EXIT_FAILURE);
    }
    // 値を初期化
    for (int i = 0; i < ACTION_NUM; i++) {
        newTree->N_arr[i] = 0;
        newTree->W_arr[i] = 0.0f;
        newTree->Q_arr[i] = 0.0f;
        newTree->children[i] = NULL;
    }
    return newTree;
}

// Treeの削除関数
void deleteTree(Tree* tree) {
    if (tree != NULL) {
        for (int i = 0; i < ACTION_NUM; i++) {
            if (tree->children[i] != NULL) {
                deleteTree(tree->children[i]);
            }
        }
        free(tree);
    }
}

// Treeに子ノードを追加する関数
void addChild(Tree* parent, int index, Tree* child) {
    if (index < 0 || index >= ACTION_NUM) {
        fprintf(stderr, "Index out of bounds\n");
        return;
    }
    parent->children[index] = child;
}

void copyIntArr(int* new_arr, int* old_arr) {
    for(int i = 0;i < ACTION_NUM;i++) new_arr[i] = old_arr[i];
}

void copyFloatArr(float* new_arr, float* old_arr, float coef) {
    for(int i = 0;i < ACTION_NUM;i++) new_arr[i] = coef * old_arr[i];
}

void multIntArr(int* new_arr, int* mult_arr) {
    for(int i = 0;i < ACTION_NUM;i++) new_arr[i] *= mult_arr[i];
}

void multFloatArr(float* new_arr, float* mult_arr) {
    for(int i = 0;i < ACTION_NUM;i++) new_arr[i] *= mult_arr[i];
}

void add_virtual_loss(Tree* tree, int action, int virtual_loss_n, int coef) {
    tree->N_arr[action] += virtual_loss_n;
    tree->W_arr[action] += coef * virtual_loss_n;  // 先後でQがひっくり返ることを考慮
    tree->Q_arr[action] = tree->W_arr[action] / tree->N_arr[action];
}

}
