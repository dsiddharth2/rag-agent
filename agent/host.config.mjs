// Host configuration for agent.
// See docs/getting-started.md for a walkthrough of each option.
export default {
  name: 'agent',
  description:
    'IBM product support assistant grounded in the TechQA technote collection.',

  agentDescription: `You answer IBM product support questions using the indexed
TechQA technotes. ALWAYS use the rag_answer tool — never answer from memory or
general knowledge.

Pass the user's full question verbatim, including error messages, codes,
versions, and product names. Default to hybrid retrieval unless the user asks
for naive retrieval.

When rag_answer returns abstained: true, tell the user clearly that the
technotes do not cover their question. When it succeeds, present the answer and
cite the technotes using the returned cited_doc_ids.`,

  fleet: {},

  comm: {
    adapter: 'express',
  },

  modules: {
    runLoop: {
      enabled: true,
      strategy: 'plan-execute',
      maxReplanAttempts: 3,
      maxReviewAttempts: 2,
      maxStepReviewAttempts: 2,
      minReviewPolicy: 'irreversible',
      maxNoActionTurns: 3,
    },
    budgets: {
      enabled: true,
      maxIterations: 25,
      maxCostUsd: 5.00,
      maxTokens: 500_000,
      timeoutMs: 600_000,
    },
    guardrails: {
      enabled: true,
      defaultPolicy: 'allow',
      validateInputs: true,
      dryRunMode: false,
    },
    dispatch: {
      enabled: true,
      store: { kind: 'sqlite', dbPath: './jobs.db' },
      concurrency: 2,
      maxQueueSize: 10,
    },
    notify: {
      sse: { enabled: true },
    },
    chat: {
      enabled: true,
      title: 'agent',
      themes: ['apra'],
    },
    router: {
      enabled: true,
      fallbackStrategy: 'plan-execute',
    },

    scheduler: {
      enabled: true,
      schedules: [
        {
          name: 'hello-every-2m',
          workflow: 'hello',
          args: {},
          cron: '*/2 * * * *',
          timezone: 'UTC',
          overlap: 'skip',
        },
      ],
    },

    // -- Memory (uncomment tiers you need) --
    //
    // memory: {
    //   // Conversation context — carries prior chat turns across tasks in a session.
    //   // The agent can reference earlier results ("book the cheapest one").
    //   conversationContext: {
    //     enabled: true,
    //     mode: 'store',                      // 'store' (server persists) or 'passthrough' (caller sends history)
    //     store: 'sqlite',
    //     dbPath: './memory/conversation.db',
    //     maxRecentTurns: 6,                  // verbatim turns in prompt
    //     maxTotalTurns: 20,                  // cap per session
    //     compactionStrategy: 'summarise',    // 'summarise' or 'sliding-window'
    //     answerMaxChars: 500,
    //   },
    //
    //   // Run state — crash recovery for interrupted tasks.
    //   runState: {
    //     enabled: true,
    //     store: 'sqlite',
    //     dbPath: './memory/run-state.db',
    //   },
    //
    //   // Long-term memory — cross-session facts with FSRS-6 decay.
    //   // Adds remember/recall/forget/promote tools to the agent.
    //   longTerm: {
    //     enabled: true,
    //     store: 'sqlite',
    //     dbPath: './memory/memory.db',
    //     autoLearn: true,                    // extract facts after each task
    //     decay: { mode: 'auto', intervalMs: 120_000 },
    //     dedup: { enabled: true },
    //     recallLimit: 20,
    //     maxEntries: 500,
    //   },
    // },
  },
};
