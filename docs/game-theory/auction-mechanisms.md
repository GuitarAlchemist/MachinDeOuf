# Auction Mechanisms

## The Problem

You run an online advertising platform. Every time a user loads a page, you have a fraction of a second to decide which ad to show. Dozens of advertisers want that slot, and each values it differently based on the user's profile. You need a mechanism that:

1. Picks the best ad efficiently.
2. Charges a fair price.
3. Gives advertisers an incentive to bid honestly (so you do not waste time on strategic gaming).

Or: a government is allocating radio spectrum licenses worth billions. Telecom companies will bid, but the auction design determines whether the outcome is efficient, whether companies overbid (winner's curse), and how much revenue the government collects.

These are all **auction design** problems -- choosing the rules that govern how bids become allocations and payments.

## The Intuition

An auction is a game. Each bidder has a private value for the item. The auction's rules determine who wins and what they pay. Different rules create different incentives:

**First-price sealed-bid:** everyone submits a bid in an envelope. Highest bid wins and pays their bid. The problem: you should bid *below* your true value (shade your bid), because paying your true value means zero profit. How much to shade? That depends on what you think others will bid -- it is a guessing game.

**Second-price sealed-bid (Vickrey):** same envelopes, but the winner pays the *second-highest* bid. Now there is no reason to shade: if you bid your true value, you either win and pay less than your value (profit!) or lose to someone who values it more (no loss). Truthful bidding is a *dominant strategy*. This is the mechanism Google uses for ad auctions.

**English (ascending):** the auctioneer raises the price; bidders drop out when it exceeds their value. The last bidder standing wins and pays just above the second-highest value. Strategically equivalent to second-price.

**All-pay:** everyone pays their bid, but only the highest bidder wins. Sounds unfair, but it models lobbying, political campaigns, and patent races where everyone expends resources regardless of outcome.

The **Revenue Equivalence Theorem** says that under standard assumptions (independent private values, risk-neutral bidders), all standard auction formats yield the same expected revenue to the seller. The differences are in risk, complexity, and strategic incentives.

## How It Works

### First-Price Auction

```
winner = bidder with highest bid
payment = winner's bid
```

**In plain English:** you pay what you bid. Optimal strategy: shade your bid below your true value by a factor of `(n-1)/n` where `n` is the number of bidders (for uniformly distributed values).

### Second-Price (Vickrey) Auction

```
winner = bidder with highest bid
payment = second-highest bid
```

**In plain English:** you pay what the runner-up bid. Optimal strategy: bid exactly your true value. This is the elegant result of the Vickrey mechanism -- truth-telling is a dominant strategy.

### English (Ascending) Auction

```
price starts low and increases by fixed increments
bidders drop out when price exceeds their value
last remaining bidder wins at current price
```

**In plain English:** the price goes up like an elevator. People get off when it passes their floor. Winner pays the floor where the second-to-last person got off.

### All-Pay Auction

```
winner = bidder with highest bid
all bidders pay their own bid (win or lose)
```

**In plain English:** everyone burns their bid amount. Only the top bidder gets anything. Models competitive spending.

## In Rust

The `ix-game` crate provides auction mechanisms:

```rust
use ix_game::auction::{
    Bid, AuctionResult,
    first_price_auction, second_price_auction, all_pay_auction,
    english_auction, dutch_auction, revenue_equivalence_test,
};

fn main() {
    // --- Create bids ---
    let bids = vec![
        Bid { bidder: 0, amount: 10.0 },
        Bid { bidder: 1, amount: 25.0 },
        Bid { bidder: 2, amount: 18.0 },
        Bid { bidder: 3, amount: 30.0 },
    ];

    // --- First-Price Sealed-Bid ---
    let fp: AuctionResult = first_price_auction(&bids).unwrap();
    println!("First-price: winner={}, payment={:.0}", fp.winner, fp.payment);
    // winner=3, payment=30 (pays own bid)

    // --- Second-Price (Vickrey) ---
    let sp: AuctionResult = second_price_auction(&bids).unwrap();
    println!("Second-price: winner={}, payment={:.0}", sp.winner, sp.payment);
    // winner=3, payment=25 (pays second-highest bid)

    // --- All-Pay ---
    let (winner, payments) = all_pay_auction(&bids).unwrap();
    println!("All-pay: winner={}, payments={:?}", winner, payments);
    // winner=3, everyone pays their bid

    // --- English (Ascending) Auction ---
    // Pass values (not strategic bids), start price, and increment.
    let values = vec![10.0, 25.0, 18.0, 30.0];
    let eng: AuctionResult = english_auction(&values, 0.0, 1.0);
    println!("English: winner={}, payment={:.0}", eng.winner, eng.payment);
    // winner=3, payment ~ 26 (just above second-highest value)

    // --- Dutch (Descending) Auction ---
    let dut: AuctionResult = dutch_auction(&values, 50.0, 1.0);
    println!("Dutch: winner={}, payment={:.0}", dut.winner, dut.payment);
    // First bidder whose value >= price wins

    // --- Revenue Equivalence Test ---
    // Compare average revenues of first-price and second-price auctions
    // over many random trials (with optimal bidding).
    let (fp_rev, sp_rev) = revenue_equivalence_test(
        5,      // 5 bidders
        10_000, // 10,000 trials
        42,     // RNG seed
    );
    println!("Average revenue: first-price={:.4}, second-price={:.4}", fp_rev, sp_rev);
    // These should be approximately equal (Revenue Equivalence Theorem).
}
```

### Key types

```rust
pub struct Bid {
    pub bidder: usize,  // Bidder identifier
    pub amount: f64,    // Bid amount
}

pub struct AuctionResult {
    pub winner: usize,       // Winning bidder's ID
    pub payment: f64,        // What the winner pays
    pub all_bids: Vec<Bid>,  // All submitted bids
}
```

### API summary

| Function | Winner determination | Payment rule | Strategic property |
|----------|--------------------|--------------|--------------------|
| `first_price_auction(&bids)` | Highest bid | Winner's bid | Shade below true value |
| `second_price_auction(&bids)` | Highest bid | Second-highest bid | Truthful bidding is dominant |
| `all_pay_auction(&bids)` | Highest bid | Everyone pays their bid | Complex equilibrium |
| `english_auction(&values, start, incr)` | Last standing | Price at second dropout | Equivalent to second-price |
| `dutch_auction(&values, start, decr)` | First to accept | Acceptance price | Equivalent to first-price |
| `revenue_equivalence_test(n, trials, seed)` | -- | -- | Empirical verification of theorem |

See the full working example: [examples/game-theory/auctions.rs](../../examples/game-theory/auctions.rs)

## When To Use This

| Scenario | Recommended Mechanism | Why |
|----------|----------------------|-----|
| Want truthful bidding (simple strategy) | Second-price / English | Dominant strategy = bid true value |
| Want to maximize revenue | Either (Revenue Equivalence) | Same expected revenue under standard assumptions |
| Bidders are risk-averse | First-price | Risk-averse bidders shade less, increasing revenue |
| Want fast resolution | Sealed-bid (first or second) | One round, no iteration |
| Want price discovery | English | Bidders observe others' willingness to pay |
| Modeling competitive spending | All-pay | Captures lobbying, R&D races |
| Multiple items | Generalize (not yet in ix-game) | VCG mechanism, combinatorial auctions |

## Key Parameters

| Parameter | Type | What it controls |
|-----------|------|-----------------|
| `bids` | `&[Bid]` | Submitted bids. For first/second-price, these are strategic bids. |
| `values` | `&[f64]` | True private values (used in English/Dutch simulations). |
| `start_price` | `f64` | Starting price for English (low) or Dutch (high) auctions. |
| `increment` / `decrement` | `f64` | Price step size. Smaller = more precise, slower simulation. |
| `num_bidders` (revenue test) | `usize` | More bidders = higher competition = higher revenue. |

## Pitfalls

1. **First-price bids are NOT true values.** If you pass true values to `first_price_auction`, you are simulating naive bidders. Rational bidders shade their bids. The `revenue_equivalence_test` function handles this correctly by applying the optimal shading factor `(n-1)/n`.

2. **English auction resolution depends on increment.** With `increment = 1.0` and values of 25 and 30, the price jumps past 25 and settles at 26. With `increment = 0.01`, it would settle at 25.01. Smaller increments are more precise but slower.

3. **Dutch auction order matters.** The implementation checks bidders in index order. The first bidder (by index) whose value exceeds the descending price wins. In a real Dutch auction, all bidders would observe the price simultaneously.

4. **All-pay auction returns payments differently.** It returns `(winner, Vec<f64>)` where the vector is indexed by bidder ID, not `AuctionResult`. This is because everyone pays, not just the winner.

5. **Revenue equivalence has assumptions.** It holds for independent private values, risk-neutral bidders, and no budget constraints. In practice, auctions behave differently: first-price generates more revenue when bidders are risk-averse, and English generates more when values are correlated.

## Going Further

- **VCG (Vickrey-Clarke-Groves) mechanism:** Generalizes second-price auctions to multiple items. Each winner pays the externality they impose on others.
- **Combinatorial auctions:** Bidders bid on *bundles* of items. Used for spectrum allocation and airport landing slots.
- **Nash equilibria in auctions:** Auction strategies form a game. See [Nash Equilibria](./nash-equilibria.md) for the general theory.
- **Mechanism design:** Design the *rules* so that the equilibrium achieves your goal (efficiency, revenue, fairness). The auction functions in ix-game are building blocks for this.
- Read: Milgrom, *Putting Auction Theory to Work* (2004) -- the practical guide by a Nobel laureate.
