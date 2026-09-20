# The Sovereign Solver — Explained Simply

*A plain-language guide to what we're building and why. No jargon. If you understand this document, you understand the project.*

---

## 0. What is this document?

The original "bible" is a detailed technical plan for engineers who already know the field. This version explains the same plan in everyday language, so anyone can understand the big decisions, why we made them, and where the risks are.

---

## 1. What are we actually building?

We're building a computer program that solves **big optimization problems** — problems like:

- "What's the cheapest way to blend crude oil into different fuel products?"
- "How should a power grid turn plants on and off to meet demand at the lowest cost?"
- "What's the best delivery route plan for a fleet of trucks?"
- "How should a factory schedule production to use the least resources?"

These problems usually have thousands or millions of variables (things you can choose, like "how much of ingredient A to use") and rules (constraints, like "can't use more of ingredient A than we have in stock"). The program needs to find the combination of choices that gets the best result (lowest cost, highest profit, etc.) while following every rule.

This kind of software is called a "solver." Famous paid ones are Gurobi and CPLEX. Well-known free ones are HiGHS, SCIP, and CBC.

### The one big rule we must follow

**We are not allowed to use any existing solver as a foundation.** We can't take HiGHS, SCIP, Gurobi's ideas wrapped in code, or any other existing solver and build on top of it. Everything that actually *solves* the problem has to be written by us, from the mathematics up.

What we *are* allowed to use: basic number-crunching tools — the equivalent of a calculator's plus/minus/multiply buttons — like libraries that do fast matrix multiplication or fast memory management. Those aren't "solvers"; they're just fast arithmetic. The rule is: if you removed all the tools we're allowed to use, what's left should still clearly be *our own* solver, not someone else's disguised.

We apply this same "would it still be ours?" test to every single piece we build, including some very advanced pieces (explained later) where it would be tempting to borrow existing free tools.

---

## 2. The big strategic decision: don't copy the old way, bet on graphics cards

### The trap we're avoiding

The "normal" way to build a solver is to write it for a regular computer processor (a CPU) — the same approach every existing solver uses. Companies like Gurobi have spent 20+ years secretly fine-tuning their CPU-based solvers. If we try to copy that approach from scratch, we will always be a worse copy of something they've already perfected. There's no way to catch up on a path where the leaders have such a massive head start.

### The bet we're making instead

We're building the solver to run mainly on a **graphics card (GPU)** instead of a regular processor (CPU). GPUs were originally built for video games and now also power AI, and they're extremely good at doing huge numbers of simple calculations all at the same time — thousands of small sums, all at once, instead of one at a time very fast.

It turns out that a certain style of optimization math — one where each step is basically "multiply a big grid of numbers together, then round a few things off" — maps perfectly onto what GPUs are good at. There's already a real-world proof this works: a supply-chain company took a giant planning problem with 30 million variables, which took a powerful CPU computer 11 minutes to solve, and solved it on a single high-end GPU in 57 seconds — with almost the exact same answer as the best commercial solver.

So our plan flips the usual roles:
- The **GPU does the heavy lifting** — the repetitive number-crunching that makes up 95% of the actual work.
- The **CPU acts like a manager** — handling the messy, irregular decision-making that doesn't parallelize well (things like "which branch of possibilities should we explore next?").

### Why this is a good bet, not just a different bet

Because almost nobody has fully built a GPU-first solver yet, there are several genuinely unsolved problems in this space — meaning if we solve them well, we're doing something new, not just re-doing something old worse. We list these unsolved problems throughout this document; they're where our actual competitive advantage comes from.

---

## 3. How the whole system fits together (the architecture)

Think of the system as a stack of floors in a building, each floor responsible for a different job. Data mostly lives on the GPU and tries to stay there as much as possible, because moving data back and forth between the CPU and GPU is slow (like shipping something between two warehouses instead of keeping it on one shelf).

From the bottom (most basic) to the top (most user-facing):

**Floor 0 — Basic number storage and math.**
This is where we decide how to store the giant grids of numbers (matrices) efficiently — mostly storing only the non-zero numbers, since most of a real-world problem's grid is empty (zeros). We also decide how precise our numbers need to be — sometimes a rough, fast estimate is fine, sometimes we need to be exact and slower.

**Floor 1 — The core number-solving engines (for the "easy" version of the problem).**
This handles problems where all the choices can be any number (fractions allowed) — no requirement that an answer be a whole number. We run **three different methods at once, racing each other**, and whichever crosses the finish line first with a valid answer wins (the others get cancelled):
- Method A: a fast, approximate GPU method (like a good enough answer very quickly).
- Method B: a slower but very precise GPU method (like taking the time to get gapless precision).
- Method C: an old-school reliable method on the regular processor, as a backup for tricky small cases.

**Floor "1.5" — The structure-finder and problem-splitter.**
Some giant problems are secretly made up of many smaller, mostly-separate sub-problems, loosely connected (imagine 50 factories that mostly operate independently but occasionally share a truck). If we can detect that shape, we can split the problem up, solve the pieces mostly independently, and combine the results — often much faster than solving the whole tangled thing at once. This floor detects that shape and, if found, hands pieces off to specialized splitting methods (explained in section 5).

**Floor 2 — Getting an exact, "clean" answer.**
The fast GPU methods above give an answer that's very close to correct but slightly fuzzy (a bit like a photo that's 99% in focus). This floor sharpens that fuzzy answer into an exact, clean, "corner-point" answer that behaves the way classic math theory expects, so later steps can trust it completely.

**Floor 3 — Simplifying the problem before solving it.**
Before solving, we cut out redundant or "obviously decided" parts of the problem — for instance, if a variable can only ever be one value given the rules, just plug in that value and remove it. This makes the problem smaller and easier before the expensive solving even begins. This floor also does the structure-detection work that feeds Floor 1.5.

**Floor 4 — The whole-number engine ("this quantity must be a whole number, not a fraction").**
Many real problems require whole numbers — you can't build half a factory or send 2.5 trucks. Handling this requires trying out different possibilities in a big decision tree (a "should we round this variable up or down?" tree, spreading out into more branches). This is the most complex floor, and it has several clever helpers built in, explained fully in section 6.

**Floor 5 — Optional AI-based improvements.**
Once the whole system works using solid, understandable math, we optionally add machine-learning "assistants" that can learn to make smarter decisions than our default rules (for example, learning which branch of the decision tree to explore first). These are optional add-ons — the system works fine without them, and we can always turn them off if they misbehave.

**Floor 6 — The front door.**
This is how people actually use the solver: feeding in a problem file, calling it from Python code, or hitting it through a web interface. No fancy visual app needed — just clean, simple ways to plug it in.

### Four rules that apply to every floor

1. **Keep data on the GPU as much as possible.** Moving data around is slow; minimize it.
2. **Always race multiple approaches at once.** Rather than picking one method and hoping it's right for this particular problem, try several simultaneously and take whichever finishes first with a correct answer.
3. **GPU proposes, CPU decides.** The GPU generates lots of candidate numbers very fast; the CPU makes the judgment calls about which ones matter.
4. **Every expensive step must earn its keep.** Cheap filtering on the CPU should always run first, so we never waste GPU time on obviously useless work.

---

## 4. The "problem-splitter" floor, explained further (Floor 1.5)

Some giant optimization problems have hidden structure — they're really several smaller, loosely-linked problems bundled together. Splitting them up and solving the pieces (mostly) separately can be much faster than solving the whole tangle at once.

We support three splitting strategies, and we build them in order of how confident we are they'll work:

1. **Method 1 (build first, most confident it'll pay off): "Guess, check, adjust."**
   Used for problems where some choices need to be made first (like "how many power plants to build") before finding out how well those choices play out across different future scenarios (like different weather patterns). We guess a first-stage decision, quickly test it against every scenario in parallel on the GPU, and if a scenario says "that guess doesn't work" or "that guess is expensive," we feed that feedback back to sharpen the guess. Repeat until it converges.

2. **Method 2 (build second, moderate confidence): "Divide into repeating chunks."**
   Used when a problem is made of many similar, mostly-independent chunks (like scheduling many similar production lines) that share just a few shared limits (like a shared warehouse). We solve a small "coordinator" problem and let the GPU rapidly generate promising chunk-level plans that satisfy the shared limits, feeding the best ones back to the coordinator.

3. **Method 3 (a "nice to have," genuinely experimental): "Whole-number version of chunk splitting."**
   This is the whole-number version of Method 2, and it's a much harder version of the same idea — hard enough that almost nobody has done it well on a GPU before. We're treating success here as a stretch goal, not a promise.

4. **Method 4 (a backup helper, not a guaranteed answer): "Average the scenarios together."**
   The simplest to make fast on a GPU, but it doesn't come with a mathematical guarantee of finding the true best answer when whole numbers are involved — we use it as a "good enough, fast" fallback rather than our main tool for those trickier cases.

**Important design idea:** splitting the problem never *replaces* trying to solve the whole thing directly — both run **at the same time**, racing each other, and whichever gets to the right answer first wins.

---

## 5. The whole-number decision tree, and its two secret weapons (Floor 4)

When a problem requires whole numbers, the solver builds a decision tree: "let's try assuming this value rounds up," "now let's try assuming it rounds down," and so on, checking each branch. Real problems can have an enormous number of possible branches, so anything that lets us skip obviously pointless branches saves enormous amounts of time. We add two advanced techniques for this:

### Secret weapon 1: Spotting "twins" in the problem (symmetry)

Many real problems have parts that are mathematically identical to each other, just relabeled — for example, if you have 5 identical delivery trucks, then "truck 1 goes to route A" and "truck 2 goes to route A, truck 1 goes elsewhere" are really the *same* decision wearing different labels. A naive solver will waste huge amounts of time separately exploring both of these "twin" possibilities, not realizing they're the same.

We build a from-scratch system that automatically spots when parts of the problem are twins of each other (using the same "detect a repeated pattern" math that's normally used to compare shapes or molecules for being structurally identical). Once we know two branches are twins, we only need to explore one of them and can skip the other entirely — sometimes cutting the number of branches to explore by nearly half.

### Secret weapon 2: Learning from mistakes (conflict learning)

When the solver tries a branch and discovers it leads to a dead end (a contradiction — the rules can't all be satisfied), a naive solver just backs up and tries something else, forgetting *why* that branch failed. Our solver instead studies *why* it failed and turns that reason into a new rule ("if we ever see this same pattern again, we already know it's a dead end — skip it immediately"). This is inspired by how modern computer chip-verification software works, adapted to this kind of math problem.

### Making sure the answer is always reproducible

Because we split work across many computer cores running at the same time, there's a risk that the exact order of operations changes slightly each time you run it (a bit like how three friends solving separate parts of a puzzle might finish in a different order each attempt) — normally that's not a bug, but it can make results maddeningly hard to double-check or debug. We add extra bookkeeping so that, no matter how many cores or what hardware you run it on, you get the exact same sequence of decisions and exact same final answer every single time. This costs us a small amount of speed but buys trustworthiness and easy debugging — something even some professional paid solvers can't fully promise.

---

## 6. Making sure the answers are actually trustworthy (numerical robustness)

This is the part the grading rubric cares about most. A solver isn't useful if it gives wrong or unstable answers on hard problems. We take several precautions:

- **Handling "confusing" problems.** Some problems have multiple equally-good answers or are set up in a way that naturally confuses simple solving methods (this is called "degeneracy"). We add tie-breaking rules and self-correcting adjustments so the solver doesn't get stuck going in circles.
- **Handling "badly scaled" problems.** If a problem mixes very large numbers (like millions of barrels) with very small ones (like fractions of a cent), rounding errors can quietly corrupt the answer. We rescale the numbers automatically before solving so everything is on a similar, well-behaved scale.
- **Handling speed-vs-precision tradeoffs.** Faster math (using less precise numbers, like a rough estimate) can sometimes go wrong on especially tricky problems. We detect when a problem looks risky and automatically fall back to slower-but-safer, fully precise math for just that part.
- **Double-checking every advanced trick.** For each of the more experimental techniques above (the problem-splitter, the reproducibility bookkeeping, etc.), we've deliberately catalogued the specific ways they could go numerically wrong and built in a specific safety net for each — see the open questions in section 9 for the ones we haven't fully solved yet.

---

## 7. How we'll know it's actually working (testing plan)

We won't just trust our own solver's answers — we test them against independent solutions:

- **Independent answer-checking.** Every answer our solver produces gets checked against a separate, independently-built solver (used only as an honest referee, never copied from) and against our own simplest from-scratch method, to catch bugs.
- **Standard test problem sets.** We test against well-known, publicly available collections of optimization problems that the whole industry uses to compare solvers — including problems specifically known to be nasty (highly repetitive/symmetric, or numerically unstable).
- **What "success" looks like.** The bar we're aiming for: solve these standard test problems reliably, and match or beat at least one existing free or paid solver on at least some of them — especially the "nasty" ones that other solvers struggle with.
- **Measuring the right things.** Besides "how long did it take," we also track things like "how many decision-tree branches did we need to check" and "how much did our twin-spotting and mistake-learning tricks actually reduce that number" — since those numbers are the direct evidence our advanced tricks are pulling their weight.

---

## 8. What makes this different from everything already out there

No single existing free solver combines all of the following, and this is our pitch for why the project is worth doing:

1. **Two very different fast-solving methods racing each other on a GPU** (instead of picking just one approach and hoping it's the right fit for every problem).
2. **A way to sharpen a fast-but-fuzzy GPU answer into a clean exact answer without losing the speed advantage** — a step that has tripped up other attempts at GPU solvers.
3. **Simplification/cleanup logic that runs on the GPU too**, instead of being a slow CPU-only step that becomes the bottleneck.
4. **AI that's trained to directly minimize total solving effort**, rather than AI trained to just imitate what a human or existing solver would do (which tends to be fragile).
5. **A shared pool of "pretty good" candidate answers** that multiple parts of the solver can borrow from and improve, rather than each part starting from scratch.
6. **Automatic problem-splitting that competes against the direct approach as standard behavior** — something no existing free solver does out of the box.
7. **A self-built "spot the twins" and "learn from mistakes" system, running reproducibly across any hardware** — a level of built-in trustworthiness that, outside of certain very expensive paid solvers, nothing else on the market currently offers.

---

## 9. Honest list of things we're not 100% sure will work

We'd rather be upfront about this than pretend everything is guaranteed. None of these risks is a dealbreaker — they're exactly the frontier we're choosing to work on, which is the whole point — but they deserve to be named plainly:

- **The "sharpen fuzzy answer into exact answer" trick** might not work equally well on every type of problem; we have a slower backup method ready if it doesn't.
- **Putting the entire whole-number decision tree onto the GPU** is still an unsolved problem industry-wide; we're only moving *parts* of it to the GPU for now.
- **Faster-but-less-precise math** could occasionally give a subtly wrong answer on the nastiest real-world problems; we watch for warning signs and fall back to fully precise (but slower) math when we see them.
- **The AI add-ons** might not generalize well to problem types they weren't trained on — we test this explicitly before trusting them.
- **Fully splitting whole-number problems into independent pieces on a GPU** (the hardest version of the "problem-splitter" idea) is something almost nobody has done — genuinely unknown whether it's even practical, which is why we treat it as a stretch goal, not a promise.
- **Our fast-approximate GPU method might not give precise enough intermediate numbers** for one of the problem-splitting techniques to work smoothly — this is a real, currently unresolved question, not something we've quietly assumed away.
- **The newest, fastest "sharpen the math" library we plan to use** hasn't been proven reliable on some of the specific, tricky number-grids our splitting methods produce; if it turns out to be unreliable there, we currently don't have a backup ready and would need to build one.
- **Deciding automatically whether a problem has "splittable structure"** is itself a hard problem, and getting it wrong (spending time looking for structure that isn't there) could actually slow things down instead of speeding them up.
- Several of the more advanced tricks (mistake-learning, twin-spotting, and the simplification step) interact with each other in ways that aren't yet fully worked out — for example, simplifying a problem too aggressively can accidentally destroy the "twin" patterns we were relying on to spot symmetry.

---

## 10. Ideas we've noted for later (not yet committed to)

These are directions that came up as promising but that we haven't fully researched or committed to building. Think of this as a running "worth looking into" list rather than a plan:

- **Speeding up the fast-approximate method further** using more advanced mathematical acceleration tricks, so it needs fewer rounds of calculation to get a usable answer.
- **Smarter, cheaper ways to estimate how "well-behaved" a problem's numbers are**, which would help us make faster and better decisions about when to fall back to slower, more precise math.
- **Extending the solver, later on, to handle problems that aren't just straight-line rules but curved/nonlinear ones** (relevant for chemical blending problems, for instance) — this would need a way to compute how a curved function changes very fast on a GPU.
- **A specific technique for tightening the "safe zone" around blending-type problems** (like crude oil blending), which are one of the named real-world use cases for this project.
- **Splitting a single enormous problem across multiple GPUs at once**, not just splitting a problem that's already logically made of separate pieces — this would let us handle problems too big to fit in one GPU's memory.
- **A lightweight "smart guesser" that predicts which solving method will likely win before racing all of them**, saving computing time by not bothering to race methods unlikely to win.

---

*End of the plain-language guide. If any section here doesn't fully make sense, the original technical bible has the complete mathematical detail — this document is meant purely as the "explain it to a smart friend" companion.*
