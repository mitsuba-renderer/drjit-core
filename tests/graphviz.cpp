#include "test.h"
#include <cstring>

TEST_CUDA(01_graphviz) {
    Float r = linspace<Float>(0, 1, 11);
    jit_var_set_label(r.index(), "r");
    jit_prefix_push(Backend, "Scope 1");
    Float a = r + 1;
    jit_var_set_label(a.index(), "a");
    jit_prefix_pop(Backend);

    jit_prefix_push(Backend, "Scope 2");
    Float b = r + 2;
    jit_var_set_label(b.index(), "b");
    jit_prefix_push(Backend, "Nested scope");
    Float c = b + 3;
    jit_var_set_label(c.index(), "c");
    Float d = a + 4;
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);
    Float e = r + 5;
    jit_var_set_label(e.index(), "e");

    jit_prefix_push(Backend, "Scope 2");
    jit_prefix_push(Backend, "Nested scope");
    Float f = a + 6;
    jit_var_set_label(f.index(), "f");
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);

    scatter_reduce(ReduceOp::Add, f, Float(4), UInt32(0));

    char *str = strdup(jit_var_graphviz());
    char *p = strstr(str, "Constant: 0x");
    if (p)
        memset(p + 12, '0', 8);

    const char *ref = R"(digraph {
    rankdir=BT;
    graph [dpi=50 fontname=Consolas];
    node [shape=record fontname=Consolas];
    edge [fontname=Consolas];
    1 [label="{mov.u32 $r0, %r0\l|{Type: cuda u32|Size: 11}|{ID #1|E:0|I:1}}}"];
    2 [label="{cvt.rn.$t0.$t1 $r0, $r1\l|{Type: cuda f32|Size: 11}|{ID #2|E:0|I:1}}}"];
    3 [label="{Constant: 0.1|{Type: cuda f32|Size: 1}|{ID #3|E:0|I:1}}}" fillcolor=gray90 style=filled];
    5 [label="{Label: \"r\"|mul.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #5|E:1|I:3}}}" fillcolor=wheat style=filled];
    subgraph cluster_5615eefd04289ffb {
        label="Scope 1";
        6 [label="{Constant: 1|{Type: cuda f32|Size: 1}|{ID #6|E:0|I:1}}}" fillcolor=gray90 style=filled];
        7 [label="{Label: \"a\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #7|E:1|I:1}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        8 [label="{Constant: 2|{Type: cuda f32|Size: 1}|{ID #8|E:0|I:1}}}" fillcolor=gray90 style=filled];
        9 [label="{Label: \"b\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #9|E:1|I:1}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            10 [label="{Constant: 3|{Type: cuda f32|Size: 1}|{ID #10|E:0|I:1}}}" fillcolor=gray90 style=filled];
            11 [label="{Label: \"c\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #11|E:1|I:0}}}" fillcolor=wheat style=filled];
            12 [label="{Constant: 4|{Type: cuda f32|Size: 1}|{ID #12|E:0|I:2}}}" fillcolor=gray90 style=filled];
            13 [label="{add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #13|E:1|I:0}}}"];
        }
    }
    14 [label="{Constant: 5|{Type: cuda f32|Size: 1}|{ID #14|E:0|I:1}}}" fillcolor=gray90 style=filled];
    15 [label="{Label: \"e\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #15|E:1|I:0}}}" fillcolor=wheat style=filled];
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            17 [label="{Label: \"f\"|Evaluated (dirty)|{Type: cuda f32|Size: 11}|{ID #17|E:1|I:1}}}" fillcolor=salmon style=filled];
        }
    }
    18 [label="{Constant: 0|{Type: cuda u32|Size: 1}|{ID #18|E:0|I:1}}}" fillcolor=gray90 style=filled];
    20 [label="{Constant: 0x000000000000|{Type: cuda ptr|Size: 1}|{ID #20|E:0|I:1}}}" fillcolor=gray90 style=filled];
    21 [label="{mad.wide.$t3 %rd3, $r3, $s2, $r1\l.reg.$t2 $r0_unused\latom.global.add.$t2 $r0_unused, [%rd3], $r2\l|{Type: cuda void |Size: 1}|{ID #21|E:1|I:0}}}" fillcolor=aquamarine style=filled];
    1 -> 2;
    2 -> 5 [label=" 1"];
    3 -> 5 [label=" 2"];
    5 -> 7 [label=" 1"];
    6 -> 7 [label=" 2"];
    5 -> 9 [label=" 1"];
    8 -> 9 [label=" 2"];
    9 -> 11 [label=" 1"];
    10 -> 11 [label=" 2"];
    7 -> 13 [label=" 1"];
    12 -> 13 [label=" 2"];
    5 -> 15 [label=" 1"];
    14 -> 15 [label=" 2"];
    17 -> 20 [label=" 4"];
    20 -> 21 [label=" 1"];
    12 -> 21 [label=" 2"];
    18 -> 21 [label=" 3"];
}
)";
}
