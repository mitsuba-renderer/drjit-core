#include "test.h"
#include <cstring>

TEST_CUDA_FP32(01_graphviz) {
    if (jit_flag(JitFlag::ForceOptiX))
        return;

    Float r = linspace<Float>(0, 1, 11);
    set_label(r, "r");
    jit_prefix_push(Backend, "Scope 1");
    Float a = r + 1;
    set_label(a, "a");
    jit_prefix_pop(Backend);

    jit_prefix_push(Backend, "Scope 2");
    Float b = r + 2;
    set_label(b, "b");
    jit_prefix_push(Backend, "Nested scope");
    Float c = b + 3;
    set_label(c, "c");
    Float d = a + 4;
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);
    Float e = r + 5;
    set_label(e, "e");

    jit_prefix_push(Backend, "Scope 2");
    jit_prefix_push(Backend, "Nested scope");
    Float f = a + 6;
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);
    Float g = Float::steal(jit_var_call_input(f.index()));

    scatter_reduce(ReduceOp::Add, f, Float(4), UInt32(0));

    char *str = strdup(jit_var_graphviz());
    char *p = strstr(str, "Literal: 0x");
    jit_assert(p);
    p += 20;
    char *p2 = strstr(p, "|");
    jit_assert(p2);
    memset(p, '0', p2-p);

    const char *ref = R"(digraph {
    rankdir=TB;
    graph [fontname=Consolas];
    node [shape=record fontname=Consolas];
    edge [fontname=Consolas];
    1 [label="{counter|{Type: cuda u32|Size: 11}|{r1|Refs: 1}}}"];
    2 [label="{cast|{Type: cuda f32|Size: 11}|{r2|Refs: 1}}}"];
    3 [label="{Literal: 0.1|{Type: cuda f32|Size: 1}|{r3|Refs: 1}}}" fillcolor=gray90 style=filled];
    subgraph cluster_5615eefd04289ffb {
        label="Scope 1";
        color=gray95;
        style=filled;
        4 [label="{Literal: 1|{Type: cuda f32|Size: 1}|{r4|Refs: 1}}}" fillcolor=gray90 style=filled];
    }
    5 [label="{Label: \"r\"|mul|{Type: cuda f32|Size: 11}|{r5|Refs: 4}}}" fillcolor=wheat style=filled];
    subgraph cluster_5615eefd04289ffb {
        label="Scope 1";
        color=gray95;
        style=filled;
        6 [label="{Label: \"a\"|add|{Type: cuda f32|Size: 11}|{r6|Refs: 3}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        color=gray95;
        style=filled;
        7 [label="{Literal: 2|{Type: cuda f32|Size: 1}|{r7|Refs: 1}}}" fillcolor=gray90 style=filled];
        8 [label="{Label: \"b\"|add|{Type: cuda f32|Size: 11}|{r8|Refs: 2}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        color=gray95;
        style=filled;
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            color=gray95;
            style=filled;
            9 [label="{Literal: 3|{Type: cuda f32|Size: 1}|{r9|Refs: 1}}}" fillcolor=gray90 style=filled];
            10 [label="{Label: \"c\"|add|{Type: cuda f32|Size: 11}|{r10|Refs: 1}}}" fillcolor=wheat style=filled];
            11 [label="{Literal: 4|{Type: cuda f32|Size: 1}|{r11|Refs: 2}}}" fillcolor=gray90 style=filled];
            12 [label="{add|{Type: cuda f32|Size: 11}|{r12|Refs: 1}}}"];
        }
    }
    13 [label="{Literal: 5|{Type: cuda f32|Size: 1}|{r13|Refs: 1}}}" fillcolor=gray90 style=filled];
    14 [label="{Label: \"e\"|add|{Type: cuda f32|Size: 11}|{r14|Refs: 1}}}" fillcolor=wheat style=filled];
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        color=gray95;
        style=filled;
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            color=gray95;
            style=filled;
            15 [label="{Literal: 6|{Type: cuda f32|Size: 1}|{r15|Refs: 1}}}" fillcolor=gray90 style=filled];
            16 [label="{add|{Type: cuda f32|Size: 11}|{r16|Refs: 1}}}"];
        }
    }
    17 [label="{bitcast|{Type: cuda f32|Size: 1}|{r17|Refs: 1}}}" fillcolor=yellow style=filled];
    18 [label="{Literal: 0|{Type: cuda u32|Size: 1}|{r18|Refs: 1}}}" fillcolor=gray90 style=filled];
    19 [label="{Literal: 1|{Type: cuda bool|Size: 1}|{r19|Refs: 1}}}" fillcolor=gray90 style=filled];
    20 [label="{Evaluated (dirty)|{Type: cuda f32|Size: 11}|{r20|Refs: 1}}}" fillcolor=salmon style=filled];
    21 [label="{Literal: 0x302000000|{Type: cuda ptr|Size: 1}|{r21|Refs: 1}}}" fillcolor=gray90 style=filled];
    22 [label="{scatter|{Type: cuda void |Size: 1}|{r22|Refs: 1}}}"];
    1 -> 2;
    2 -> 5 [label=" 1"];
    3 -> 5 [label=" 2"];
    5 -> 6 [label=" 1"];
    4 -> 6 [label=" 2"];
    5 -> 8 [label=" 1"];
    7 -> 8 [label=" 2"];
    8 -> 10 [label=" 1"];
    9 -> 10 [label=" 2"];
    6 -> 12 [label=" 1"];
    11 -> 12 [label=" 2"];
    5 -> 14 [label=" 1"];
    13 -> 14 [label=" 2"];
    6 -> 16 [label=" 1"];
    15 -> 16 [label=" 2"];
    16 -> 17;
    20 -> 21 [style=dashed];
    21 -> 22 [label=" 1"];
    11 -> 22 [label=" 2"];
    18 -> 22 [label=" 3"];
    19 -> 22 [label=" 4"];
    subgraph cluster_legend {
        label="Legend";
        l5 [style=filled fillcolor=yellow label="Symbolic"];
        l4 [style=filled fillcolor=yellowgreen label="Special"];
        l3 [style=filled fillcolor=salmon label="Dirty"];
        l2 [style=filled fillcolor=lightblue2 label="Evaluated"];
        l1 [style=filled fillcolor=wheat label="Labeled"];
        l0 [style=filled fillcolor=gray90 label="Constant"];
    }
}
)";
    if (strcmp(ref, str) != 0) {
        FILE *f1 = fopen("a.txt", "wb");
        FILE *f2 = fopen("b.txt", "wb");
        fputs(ref, f1);
        fputs(str, f2);
        fclose(f1);
        fclose(f2);
    }
    jit_assert(strcmp(ref, str) == 0);
}
