# What Is Machine Learning?

## 1. The Classical Programming Paradigm

In classical programming, a developer writes explicit rules that tell the computer what to do.

The usual flow is:

```text
rules + data -> output
```

For example, imagine a program that decides whether a customer gets a discount:

```text
if customer_age > 60:
    apply senior discount
elif customer_has_coupon:
    apply coupon discount
else:
    no discount
```

This works well when the problem can be described with clear rules. The programmer understands the logic, writes the logic, and the computer applies it.

**Simple intuition:** classical programming is rule-driven. Humans define the behavior, and the machine executes it.

---

## 2. Visible Limitations of Classical Programming

Classical programming becomes harder when the rules are difficult to write explicitly.

Some problems have patterns that are real but not easy to describe with hand-written conditions:

- recognizing a face in an image;
- understanding whether an email is spam;
- predicting house prices;
- translating text from one language to another;
- detecting fraud in financial transactions;
- recommending products, songs, or movies.

For these problems, the decision logic may depend on thousands or millions of small signals. A human can often recognize the pattern, but writing exact rules for it is extremely difficult.

For example, how would we write all the rules that define what makes an image contain a cat?

```text
if ears are triangular?
if eyes have a certain shape?
if fur texture is visible?
if body position is unusual?
if lighting changes?
if the cat is partly hidden?
```

The rules quickly become incomplete, fragile, and impossible to maintain.

**Main limitation:** classical programming requires us to know the rules in advance, but many real-world problems involve patterns that are too complex, noisy, or uncertain to define manually.

---

## 3. What Is Machine Learning?

Machine Learning changes the paradigm.

Instead of manually writing all the rules, we give the machine examples and let it learn patterns from data.

The Machine Learning flow is:

```text
data + expected outputs -> learned model
```

Then, once the model has learned from examples, we use it to make predictions:

```text
new data + learned model -> prediction
```

A common definition is:

> Machine Learning is a field of computer science that studies algorithms able to learn patterns from data and use those patterns to make predictions, decisions, or representations without being explicitly programmed for every case.

The important idea is not that the machine "understands" like a human. The important idea is that the model adjusts its internal parameters so that its predictions become better according to a chosen objective.

**Simple intuition:** Machine Learning is pattern learning from examples.

**Engineer note:** in practice, learning means optimizing model parameters to reduce a loss function on data, while trying to generalize to unseen data.

---

## 4. Why Probability and Statistics Matter

Machine Learning learns from data, but data is never the full reality. It is a finite sample from a larger and uncertain process.

This is why probability and statistics are foundational.

Probability helps us describe the uncertain process that could generate the data

Statistics helps us learn about that process from observed samples

Together:

> Probability models the world behind the data. Statistics learns about that world from the data we observe.

This connection is central to Machine Learning:

- data is treated as a sample from an unknown distribution;
- labels and features can be modeled as random variables;
- learning objectives are often written as expected losses;
- training uses finite samples to estimate useful patterns;
- evaluation tries to measure whether the model generalizes beyond the sample.
