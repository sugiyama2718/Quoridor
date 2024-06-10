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
    for (int i = 0; i < 137; i++) {
        newTree->N[i] = 0;
        newTree->children[i] = NULL;
    }
    return newTree;
}

// Treeの削除関数
void deleteTree(Tree* tree) {
    if (tree != NULL) {
        for (int i = 0; i < 137; i++) {
            if (tree->children[i] != NULL) {
                deleteTree(tree->children[i]);
            }
        }
        free(tree);
    }
}

// Treeに子ノードを追加する関数
void addChild(Tree* parent, int index, Tree* child) {
    if (index < 0 || index >= 137) {
        fprintf(stderr, "Index out of bounds\n");
        return;
    }
    parent->children[index] = child;
}

// Treeの値を設定する関数
void setTreeValue(Tree* tree, int index, int value) {
    if (index < 0 || index >= 137) {
        fprintf(stderr, "Index out of bounds\n");
        return;
    }
    tree->N[index] = value;
}

// Treeの値を取得する関数
int getTreeValue(Tree* tree, int index) {
    if (index < 0 || index >= 137) {
        fprintf(stderr, "Index out of bounds\n");
        return -1; // エラー値
    }
    return tree->N[index];
}

}
