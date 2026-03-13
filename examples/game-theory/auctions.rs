//! Run Auctions
//!
//! Simulate different auction mechanisms and compare outcomes.
//!
//! ```bash
//! cargo run --example auctions
//! ```

use machin_game::auction::{
    dutch_auction, english_auction, first_price_auction, second_price_auction,
};

fn main() {
    let bids = vec![10.0, 25.0, 18.0, 30.0, 22.0];
    println!("Bids: {:?}\n", bids);

    let (winner_fp, price_fp) = first_price_auction(&bids);
    println!(
        "First-price:  winner=bidder {}, pays={:.0}",
        winner_fp, price_fp
    );

    let (winner_sp, price_sp) = second_price_auction(&bids);
    println!(
        "Second-price: winner=bidder {}, pays={:.0}",
        winner_sp, price_sp
    );

    let (winner_en, price_en) = english_auction(&bids, 5.0, 1.0);
    println!(
        "English:      winner=bidder {}, pays={:.0}",
        winner_en, price_en
    );

    let (winner_du, price_du) = dutch_auction(&bids, 50.0, 1.0);
    println!(
        "Dutch:        winner=bidder {}, pays={:.0}",
        winner_du, price_du
    );
}
