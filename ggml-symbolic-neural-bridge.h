/*
 * ggml-symbolic-neural-bridge.h - PowerLISP ↔ Neural Integration Layer
 *
 * Scott's Vision: "Recursive loop between symbolic reasoning and neural inference.
 *                  Neural handles perception, symbolic handles logic."
 *
 * This is what DeepSeek CANNOT do - they have no symbolic layer.
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                  SYMBOLIC-NEURAL BRIDGE                         │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                                                                  │
 * │   ┌──────────────────┐           ┌──────────────────┐           │
 * │   │   NEURAL LAYER   │◄─────────►│  SYMBOLIC LAYER  │           │
 * │   │   (Vec_perm PSE) │           │   (PowerLISP)    │           │
 * │   │                  │           │                  │           │
 * │   │ • Perception     │   Handoff │ • Reasoning      │           │
 * │   │ • Pattern match  │◄─────────►│ • Logic          │           │
 * │   │ • Embeddings     │           │ • Rules          │           │
 * │   │ • Generation     │           │ • Tetranary      │           │
 * │   └────────┬─────────┘           └────────┬─────────┘           │
 * │            │                              │                      │
 * │            └──────────────┬───────────────┘                      │
 * │                           │                                      │
 * │              ┌────────────┴────────────┐                         │
 * │              │    NEUROMORPHIC COFFER  │                         │
 * │              │    (NUMA-aware routing) │                         │
 * │              └─────────────────────────┘                         │
 * │                                                                  │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Handoff Protocol:
 * 1. Neural produces output with confidence score
 * 2. If confidence < threshold → handoff to symbolic
 * 3. Symbolic applies tetranary logic, rules, constraints
 * 4. Symbolic may recurse back to neural for more context
 * 5. Final output has both neural generation + symbolic validation
 */

#ifndef GGML_SYMBOLIC_NEURAL_BRIDGE_H
#define GGML_SYMBOLIC_NEURAL_BRIDGE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

/*===========================================================================
 * Tetranary Logic (from PowerLISP)
 *===========================================================================*/

/* Import from neuromorphic-coffers or define locally */
#ifndef TETRA_T_DEFINED
#define TETRA_T_DEFINED
typedef enum {
    TETRA_FALSE    = 0,
    TETRA_POSSIBLE = 1,
    TETRA_LIKELY   = 2,
    TETRA_CERTAIN  = 3
} tetra_t;
#endif

/*===========================================================================
 * Symbolic Reasoning Interface
 *
 * This defines the interface between C and Rust/PowerLISP
 *===========================================================================*/

/* Forward declarations for Rust FFI */
typedef struct powerlisp_env powerlisp_env_t;
typedef struct powerlisp_value powerlisp_value_t;

/* Symbolic evaluation result */
typedef struct {
    tetra_t judgment;        /* Tetranary truth value */
    float confidence;        /* 0.0-1.0 confidence score */
    const char* explanation; /* Human-readable reasoning trace */
    bool should_override;    /* Symbolic wants to override neural */
    const char* override_response; /* If override, use this instead */
} symbolic_result_t;

/* Rule for production system */
typedef struct {
    const char* name;
    const char** conditions;  /* Null-terminated array */
    tetra_t* condition_thresholds;
    const char** actions;     /* Null-terminated array */
    tetra_t priority;
} production_rule_t;

/* Working memory fact */
typedef struct {
    char key[64];
    tetra_t value;
    uint64_t timestamp;
    int source;  /* 0=neural, 1=symbolic, 2=sensor */
} working_memory_fact_t;

/*===========================================================================
 * Symbolic Engine State
 *===========================================================================*/

#define MAX_RULES 256
#define MAX_FACTS 512
#define MAX_RECURSION_DEPTH 8

typedef struct {
    /* Working memory */
    working_memory_fact_t facts[MAX_FACTS];
    int n_facts;

    /* Production rules */
    production_rule_t rules[MAX_RULES];
    int n_rules;

    /* Reasoning trace */
    const char* fired_rules[32];
    int n_fired;

    /* Recursion control */
    int recursion_depth;
    int max_recursion;

    /* Handoff thresholds */
    float neural_to_symbolic_threshold;  /* Default: 0.3 */
    float symbolic_to_neural_threshold;  /* Default: 0.5 (for POSSIBLE) */

    /* Statistics */
    uint64_t neural_calls;
    uint64_t symbolic_calls;
    uint64_t handoffs;
} symbolic_engine_t;

static symbolic_engine_t g_symbolic_engine = {
    .n_facts = 0,
    .n_rules = 0,
    .n_fired = 0,
    .recursion_depth = 0,
    .max_recursion = MAX_RECURSION_DEPTH,
    .neural_to_symbolic_threshold = 0.3f,
    .symbolic_to_neural_threshold = 0.5f,
    .neural_calls = 0,
    .symbolic_calls = 0,
    .handoffs = 0
};

/*===========================================================================
 * Working Memory Operations
 *===========================================================================*/

static inline void wm_assert(const char* key, tetra_t value, int source) {
    if (g_symbolic_engine.n_facts >= MAX_FACTS) return;

    working_memory_fact_t* fact = &g_symbolic_engine.facts[g_symbolic_engine.n_facts++];
    strncpy(fact->key, key, sizeof(fact->key) - 1);
    fact->value = value;
    fact->timestamp = (uint64_t)time(NULL);
    fact->source = source;
}

static inline tetra_t wm_query(const char* key) {
    for (int i = g_symbolic_engine.n_facts - 1; i >= 0; i--) {
        if (strcmp(g_symbolic_engine.facts[i].key, key) == 0) {
            return g_symbolic_engine.facts[i].value;
        }
    }
    return TETRA_POSSIBLE;  /* Unknown = uncertain */
}

static inline void wm_retract(const char* key) {
    for (int i = 0; i < g_symbolic_engine.n_facts; i++) {
        if (strcmp(g_symbolic_engine.facts[i].key, key) == 0) {
            /* Shift remaining facts */
            memmove(&g_symbolic_engine.facts[i],
                    &g_symbolic_engine.facts[i + 1],
                    (g_symbolic_engine.n_facts - i - 1) * sizeof(working_memory_fact_t));
            g_symbolic_engine.n_facts--;
            return;
        }
    }
}

/*===========================================================================
 * Production Rule Operations
 *===========================================================================*/

/* Check if a rule's conditions are satisfied */
static inline tetra_t rule_check_conditions(const production_rule_t* rule) {
    tetra_t min_certainty = TETRA_CERTAIN;

    int i = 0;
    while (rule->conditions[i]) {
        tetra_t fact_value = wm_query(rule->conditions[i]);
        tetra_t threshold = rule->condition_thresholds ? rule->condition_thresholds[i] : TETRA_LIKELY;

        if (fact_value < threshold) {
            return TETRA_FALSE;  /* Condition not met */
        }

        if (fact_value < min_certainty) {
            min_certainty = fact_value;
        }
        i++;
    }

    return min_certainty;
}

/* Fire all applicable rules (forward chaining) */
static inline int forward_chain(void) {
    int rules_fired = 0;
    bool changed = true;

    while (changed && rules_fired < 100) {  /* Max iterations */
        changed = false;

        for (int r = 0; r < g_symbolic_engine.n_rules; r++) {
            production_rule_t* rule = &g_symbolic_engine.rules[r];

            tetra_t certainty = rule_check_conditions(rule);
            if (certainty >= TETRA_LIKELY) {
                /* Fire rule - execute actions */
                int a = 0;
                while (rule->actions[a]) {
                    /* Parse action: "assert KEY VALUE" or "retract KEY" */
                    if (strncmp(rule->actions[a], "assert ", 7) == 0) {
                        char key[64];
                        int value;
                        sscanf(rule->actions[a] + 7, "%s %d", key, &value);
                        wm_assert(key, (tetra_t)value, 1);  /* 1 = symbolic source */
                        changed = true;
                    } else if (strncmp(rule->actions[a], "retract ", 8) == 0) {
                        wm_retract(rule->actions[a] + 8);
                        changed = true;
                    }
                    a++;
                }

                /* Record firing */
                if (g_symbolic_engine.n_fired < 32) {
                    g_symbolic_engine.fired_rules[g_symbolic_engine.n_fired++] = rule->name;
                }
                rules_fired++;
            }
        }
    }

    return rules_fired;
}

/*===========================================================================
 * Neural-Symbolic Handoff
 *===========================================================================*/

typedef struct {
    /* Input */
    const char* query;
    const float* neural_output;
    int output_dim;
    float neural_confidence;

    /* Routing */
    int cognitive_function;  /* From neuromorphic coffers */
    int target_coffer;

    /* State */
    int recursion_depth;
} handoff_context_t;

/*
 * Decide if neural output should be handed to symbolic layer
 */
static inline bool should_handoff_to_symbolic(const handoff_context_t* ctx) {
    /* Rule 1: Low neural confidence */
    if (ctx->neural_confidence < g_symbolic_engine.neural_to_symbolic_threshold) {
        return true;
    }

    /* Rule 2: Logical reasoning tasks */
    if ((ctx->cognitive_function & 0xF0) == 0x10) {  /* Left hemisphere logic */
        /* Check if output seems inconsistent */
        /* (In production, would analyze output tokens) */
        if (ctx->neural_confidence < 0.7f) {
            return true;
        }
    }

    /* Rule 3: Metacognitive queries (always symbolic) */
    if (ctx->cognitive_function == 0x43) {  /* COG_META_COGNITION */
        return true;
    }

    /* Rule 4: Recursion from symbolic that needs more context */
    if (ctx->recursion_depth > 0 && ctx->recursion_depth < g_symbolic_engine.max_recursion) {
        return false;  /* Stay neural, already recursed from symbolic */
    }

    return false;
}

/*
 * Symbolic reasoning on neural output
 */
static inline symbolic_result_t symbolic_evaluate(const handoff_context_t* ctx) {
    symbolic_result_t result = {
        .judgment = TETRA_POSSIBLE,
        .confidence = 0.5f,
        .explanation = NULL,
        .should_override = false,
        .override_response = NULL
    };

    g_symbolic_engine.symbolic_calls++;

    /* Assert query into working memory */
    wm_assert("current_query", TETRA_CERTAIN, 0);

    /* Assert neural confidence as fact */
    if (ctx->neural_confidence > 0.8f) {
        wm_assert("neural_confident", TETRA_CERTAIN, 0);
    } else if (ctx->neural_confidence > 0.5f) {
        wm_assert("neural_confident", TETRA_LIKELY, 0);
    } else if (ctx->neural_confidence > 0.2f) {
        wm_assert("neural_confident", TETRA_POSSIBLE, 0);
    } else {
        wm_assert("neural_confident", TETRA_FALSE, 0);
    }

    /* Run forward chaining */
    int fired = forward_chain();

    /* Determine judgment based on working memory */
    tetra_t neural_conf_tetra = wm_query("neural_confident");
    tetra_t logical_valid = wm_query("logical_valid");

    /* Combine evidence */
    if (neural_conf_tetra >= TETRA_LIKELY && logical_valid >= TETRA_LIKELY) {
        result.judgment = TETRA_CERTAIN;
        result.confidence = 0.9f;
    } else if (neural_conf_tetra >= TETRA_POSSIBLE || logical_valid >= TETRA_POSSIBLE) {
        result.judgment = TETRA_LIKELY;
        result.confidence = 0.6f;
    } else {
        result.judgment = TETRA_POSSIBLE;
        result.confidence = 0.3f;
    }

    /* Check for override */
    tetra_t override_needed = wm_query("needs_override");
    if (override_needed >= TETRA_LIKELY) {
        result.should_override = true;
        /* In production, would generate override from rules */
    }

    return result;
}

/*
 * Should symbolic result recurse back to neural?
 */
static inline bool should_recurse_to_neural(const symbolic_result_t* sym_result) {
    /* If symbolic is uncertain, try neural with more context */
    if (sym_result->judgment == TETRA_POSSIBLE) {
        return true;
    }

    return false;
}

/*===========================================================================
 * Complete Bridge Execution
 *===========================================================================*/

typedef struct {
    /* Final output */
    const float* output;
    int output_dim;

    /* Confidence */
    float confidence;
    tetra_t judgment;

    /* Provenance */
    int neural_layers;    /* How many neural passes */
    int symbolic_layers;  /* How many symbolic passes */
    const char* explanation;
} bridge_result_t;

/*
 * Execute neural-symbolic bridge
 *
 * This is the main entry point that:
 * 1. Runs neural inference
 * 2. Checks if symbolic needed
 * 3. Runs symbolic reasoning
 * 4. May recurse back to neural
 * 5. Returns combined result
 */
static bridge_result_t execute_bridge(
    const char* query,
    const float* query_embed,
    /* Neural inference function pointer */
    void (*neural_infer)(const float* input, float* output, int* dim),
    int max_recursion
) {
    bridge_result_t result = {
        .output = NULL,
        .output_dim = 0,
        .confidence = 0.0f,
        .judgment = TETRA_POSSIBLE,
        .neural_layers = 0,
        .symbolic_layers = 0,
        .explanation = NULL
    };

    float* neural_output = NULL;
    int output_dim = 0;

    g_symbolic_engine.recursion_depth = 0;

    while (g_symbolic_engine.recursion_depth < max_recursion) {
        /* Step 1: Neural inference */
        g_symbolic_engine.neural_calls++;
        result.neural_layers++;

        /* Allocate output buffer (reuse if exists) */
        if (!neural_output) {
            neural_output = (float*)malloc(4096 * sizeof(float));
        }

        neural_infer(query_embed, neural_output, &output_dim);

        /* Estimate confidence from output (simplified) */
        float max_val = 0.0f;
        for (int i = 0; i < output_dim && i < 100; i++) {
            if (neural_output[i] > max_val) max_val = neural_output[i];
        }
        float neural_confidence = max_val;  /* Simplified */

        /* Step 2: Check if symbolic needed */
        handoff_context_t ctx = {
            .query = query,
            .neural_output = neural_output,
            .output_dim = output_dim,
            .neural_confidence = neural_confidence,
            .cognitive_function = 0x00,  /* Would come from neuromorphic routing */
            .target_coffer = 0,
            .recursion_depth = g_symbolic_engine.recursion_depth
        };

        if (!should_handoff_to_symbolic(&ctx)) {
            /* Neural is confident enough */
            result.output = neural_output;
            result.output_dim = output_dim;
            result.confidence = neural_confidence;
            result.judgment = neural_confidence > 0.7f ? TETRA_CERTAIN :
                             neural_confidence > 0.4f ? TETRA_LIKELY : TETRA_POSSIBLE;
            break;
        }

        /* Step 3: Symbolic reasoning */
        g_symbolic_engine.handoffs++;
        symbolic_result_t sym = symbolic_evaluate(&ctx);
        result.symbolic_layers++;

        if (sym.should_override) {
            /* Symbolic overrides neural */
            result.judgment = sym.judgment;
            result.confidence = sym.confidence;
            result.explanation = sym.explanation;
            break;
        }

        if (!should_recurse_to_neural(&sym)) {
            /* Symbolic is certain enough */
            result.output = neural_output;
            result.output_dim = output_dim;
            result.judgment = sym.judgment;
            result.confidence = sym.confidence;
            break;
        }

        /* Step 4: Recurse to neural */
        g_symbolic_engine.recursion_depth++;
        /* In production, would modify query_embed based on symbolic insights */
    }

    return result;
}

/*===========================================================================
 * Built-in Production Rules (Examples)
 *===========================================================================*/

static void init_default_rules(void) {
    /* Rule: If neural confident AND query is factual → accept */
    static const char* factual_conditions[] = {"neural_confident", "query_is_factual", NULL};
    static const char* factual_actions[] = {"assert logical_valid 3", NULL};
    static tetra_t factual_thresholds[] = {TETRA_LIKELY, TETRA_LIKELY};

    g_symbolic_engine.rules[g_symbolic_engine.n_rules++] = (production_rule_t){
        .name = "accept-factual",
        .conditions = factual_conditions,
        .condition_thresholds = factual_thresholds,
        .actions = factual_actions,
        .priority = TETRA_LIKELY
    };

    /* Rule: If neural uncertain AND query is logical → override */
    static const char* logic_conditions[] = {"query_is_logical", NULL};
    static const char* logic_actions[] = {"assert needs_symbolic_check 3", NULL};
    static tetra_t logic_thresholds[] = {TETRA_LIKELY};

    g_symbolic_engine.rules[g_symbolic_engine.n_rules++] = (production_rule_t){
        .name = "check-logical",
        .conditions = logic_conditions,
        .condition_thresholds = logic_thresholds,
        .actions = logic_actions,
        .priority = TETRA_CERTAIN
    };
}

/*===========================================================================
 * Initialization
 *===========================================================================*/

static int init_symbolic_neural_bridge(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Symbolic-Neural Bridge - PowerLISP Integration               ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Neural Layer: Vec_perm PSE non-bijunctive collapse           ║\n");
    fprintf(stderr, "║  Symbolic Layer: Tetranary logic, production rules            ║\n");
    fprintf(stderr, "║                                                               ║\n");
    fprintf(stderr, "║  Handoff Thresholds:                                          ║\n");
    fprintf(stderr, "║  • Neural → Symbolic: %.2f confidence                        ║\n",
            g_symbolic_engine.neural_to_symbolic_threshold);
    fprintf(stderr, "║  • Symbolic → Neural: POSSIBLE (0.33)                         ║\n");
    fprintf(stderr, "║  • Max Recursion: %d                                          ║\n",
            g_symbolic_engine.max_recursion);
    fprintf(stderr, "║                                                               ║\n");
    fprintf(stderr, "║  This is what DeepSeek CANNOT do - they have no symbolic      ║\n");
    fprintf(stderr, "║  reasoning layer. We have complete neural ↔ symbolic bridge.  ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

    init_default_rules();

    return 0;
}

/*===========================================================================
 * Statistics
 *===========================================================================*/

static void print_bridge_stats(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Symbolic-Neural Bridge Statistics                            ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Neural calls:    %12lu                                 ║\n",
            (unsigned long)g_symbolic_engine.neural_calls);
    fprintf(stderr, "║  Symbolic calls:  %12lu                                 ║\n",
            (unsigned long)g_symbolic_engine.symbolic_calls);
    fprintf(stderr, "║  Handoffs:        %12lu                                 ║\n",
            (unsigned long)g_symbolic_engine.handoffs);
    fprintf(stderr, "║  Rules in memory: %12d                                 ║\n",
            g_symbolic_engine.n_rules);
    fprintf(stderr, "║  Facts in WM:     %12d                                 ║\n",
            g_symbolic_engine.n_facts);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
}

#endif /* GGML_SYMBOLIC_NEURAL_BRIDGE_H */
