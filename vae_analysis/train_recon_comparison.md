# Enhanced VAE Reconstruction Comparison

_Generated: 2025-08-27T13:03:43_

This analysis includes the following reconstructions:
- **Ego View**: Character, color, and class predictions in ego-centric window
- **Bag Elements**: High-probability glyph elements
- **Passability/Safety**: 3x3 grids around hero position

## Sample 1

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 0](images/train_sample_000_orig.png) | ![recon 0](images/train_sample_000_recon.png) |

**Accuracy**: Character: 0.438, Color: 0.612

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 0](images/train_sample_000_ego_class_orig.png) | ![recon class 0](images/train_sample_000_ego_class_recon.png) |

**Class Accuracy**: 0.620

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (17 items):
------------------------------
  '#' (color  7)
  '%' (color  3)
  '(' (color  3)
  ')' (color  3)
  ')' (color  6)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '>' (color  7)
  '@' (color 15)
  '[' (color  8)
  '_' (color  7)
  '`' (color  7)
  'r' (color  3)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (13 items):
------------------------------
  '#' (color  7)
  ')' (color  6)
  '*' (color  8)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '@' (color 15)
  '[' (color  3)
  '`' (color  7)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 10 items
    '#' (color  7)
    ')' (color  6)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '@' (color 15)
    '`' (color  7)
    '|' (color  3)
    '|' (color  7)
  Missed items: 7 items
    '%' (color  3)
    '(' (color  3)
    ')' (color  3)
    '>' (color  7)
    '[' (color  8)
    '_' (color  7)
    'r' (color  3)
  False positives: 3 items
    '*' (color  8)
    '<' (color  7)
    '[' (color  3)

Performance Summary:
------------------------------
  Precision: 0.769 (10/13)
  Recall: 0.588 (10/17)
  F1-Score: 0.667
  Total unique items: 20
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 0](images/train_sample_000_pass_safe_orig.png) | ![recon pass safe 0](images/train_sample_000_pass_safe_recon.png) |

================================================================================

## Sample 2

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 1](images/train_sample_001_orig.png) | ![recon 1](images/train_sample_001_recon.png) |

**Accuracy**: Character: 0.273, Color: 0.661

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 1](images/train_sample_001_ego_class_orig.png) | ![recon class 1](images/train_sample_001_ego_class_recon.png) |

**Class Accuracy**: 0.529

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (14 items):
------------------------------
  '#' (color  7)
  ')' (color  6)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '_' (color  7)
  'd' (color 15)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (14 items):
------------------------------
  '#' (color  7)
  '+' (color  3)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  'd' (color 15)
  'f' (color 15)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 11 items
    '#' (color  7)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '>' (color  7)
    '@' (color 15)
    'd' (color 15)
    '|' (color  3)
    '|' (color  7)
  Missed items: 3 items
    ')' (color  6)
    '_' (color  7)
    '{' (color 12)
  False positives: 3 items
    '+' (color  3)
    '`' (color  7)
    'f' (color 15)

Performance Summary:
------------------------------
  Precision: 0.786 (11/14)
  Recall: 0.786 (11/14)
  F1-Score: 0.786
  Total unique items: 17
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 1](images/train_sample_001_pass_safe_orig.png) | ![recon pass safe 1](images/train_sample_001_pass_safe_recon.png) |

================================================================================

## Sample 3

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 2](images/train_sample_002_orig.png) | ![recon 2](images/train_sample_002_recon.png) |

**Accuracy**: Character: 0.174, Color: 0.620

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 2](images/train_sample_002_ego_class_orig.png) | ![recon class 2](images/train_sample_002_ego_class_recon.png) |

**Class Accuracy**: 0.653

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (17 items):
------------------------------
  '#' (color  7)
  '%' (color  3)
  '%' (color 10)
  ')' (color  6)
  '*' (color  1)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '@' (color 15)
  'F' (color 10)
  '[' (color  6)
  '^' (color  8)
  '`' (color  7)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (16 items):
------------------------------
  '#' (color  7)
  ')' (color  6)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  ':' (color 15)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '^' (color 12)
  '`' (color  7)
  'd' (color 15)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 11 items
    '#' (color  7)
    ')' (color  6)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '@' (color 15)
    '`' (color  7)
    '|' (color  3)
    '|' (color  7)
  Missed items: 6 items
    '%' (color  3)
    '%' (color 10)
    '*' (color  1)
    'F' (color 10)
    '[' (color  6)
    '^' (color  8)
  False positives: 5 items
    ':' (color 15)
    '>' (color  7)
    '^' (color 12)
    'd' (color 15)
    '{' (color 12)

Performance Summary:
------------------------------
  Precision: 0.688 (11/16)
  Recall: 0.647 (11/17)
  F1-Score: 0.667
  Total unique items: 22
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 2](images/train_sample_002_pass_safe_orig.png) | ![recon pass safe 2](images/train_sample_002_pass_safe_recon.png) |

================================================================================

## Sample 4

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 3](images/train_sample_003_orig.png) | ![recon 3](images/train_sample_003_recon.png) |

**Accuracy**: Character: 0.306, Color: 0.612

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 3](images/train_sample_003_ego_class_orig.png) | ![recon class 3](images/train_sample_003_ego_class_recon.png) |

**Class Accuracy**: 0.579

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (24 items):
------------------------------
  '#' (color  7)
  '%' (color  3)
  '(' (color  3)
  ')' (color  3)
  ')' (color  6)
  '*' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '[' (color  8)
  '^' (color  3)
  '^' (color  8)
  '^' (color 12)
  '_' (color  7)
  '`' (color  7)
  'd' (color 15)
  'o' (color 15)
  '|' (color  3)
  '|' (color  7)
  '|' (color 15)

Reconstructed Bag (17 items):
------------------------------
  '#' (color  7)
  '%' (color  3)
  '(' (color  3)
  ')' (color  6)
  '*' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  'd' (color 15)
  'f' (color 15)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 16 items
    '#' (color  7)
    '%' (color  3)
    '(' (color  3)
    ')' (color  6)
    '*' (color  7)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '>' (color  7)
    '@' (color 15)
    '`' (color  7)
    'd' (color 15)
    '|' (color  3)
    '|' (color  7)
  Missed items: 8 items
    ')' (color  3)
    '[' (color  8)
    '^' (color  3)
    '^' (color  8)
    '^' (color 12)
    '_' (color  7)
    'o' (color 15)
    '|' (color 15)
  False positives: 1 items
    'f' (color 15)

Performance Summary:
------------------------------
  Precision: 0.941 (16/17)
  Recall: 0.667 (16/24)
  F1-Score: 0.780
  Total unique items: 25
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 3](images/train_sample_003_pass_safe_orig.png) | ![recon pass safe 3](images/train_sample_003_pass_safe_recon.png) |

================================================================================

## Sample 5

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 4](images/train_sample_004_orig.png) | ![recon 4](images/train_sample_004_recon.png) |

**Accuracy**: Character: 0.231, Color: 0.488

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 4](images/train_sample_004_ego_class_orig.png) | ![recon class 4](images/train_sample_004_ego_class_recon.png) |

**Class Accuracy**: 0.430

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (14 items):
------------------------------
  '#' (color  6)
  '#' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  'f' (color 15)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (11 items):
------------------------------
  '#' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  'd' (color 15)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 10 items
    '#' (color  7)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '>' (color  7)
    '@' (color 15)
    '|' (color  3)
    '|' (color  7)
  Missed items: 4 items
    '#' (color  6)
    '`' (color  7)
    'f' (color 15)
    '{' (color 12)
  False positives: 1 items
    'd' (color 15)

Performance Summary:
------------------------------
  Precision: 0.909 (10/11)
  Recall: 0.714 (10/14)
  F1-Score: 0.800
  Total unique items: 15
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 4](images/train_sample_004_pass_safe_orig.png) | ![recon pass safe 4](images/train_sample_004_pass_safe_recon.png) |

================================================================================

## Sample 6

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 5](images/train_sample_005_orig.png) | ![recon 5](images/train_sample_005_recon.png) |

**Accuracy**: Character: 0.273, Color: 0.620

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 5](images/train_sample_005_ego_class_orig.png) | ![recon class 5](images/train_sample_005_ego_class_recon.png) |

**Class Accuracy**: 0.562

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (16 items):
------------------------------
  '#' (color  7)
  '*' (color  8)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '[' (color  3)
  '[' (color  8)
  '^' (color  6)
  '`' (color  7)
  'd' (color 15)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (11 items):
------------------------------
  '#' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '@' (color 15)
  '[' (color  3)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 10 items
    '#' (color  7)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '@' (color 15)
    '[' (color  3)
    '|' (color  3)
    '|' (color  7)
  Missed items: 6 items
    '*' (color  8)
    '>' (color  7)
    '[' (color  8)
    '^' (color  6)
    '`' (color  7)
    'd' (color 15)
  False positives: 1 items
    '{' (color 12)

Performance Summary:
------------------------------
  Precision: 0.909 (10/11)
  Recall: 0.625 (10/16)
  F1-Score: 0.741
  Total unique items: 17
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 5](images/train_sample_005_pass_safe_orig.png) | ![recon pass safe 5](images/train_sample_005_pass_safe_recon.png) |

================================================================================

## Sample 7

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 6](images/train_sample_006_orig.png) | ![recon 6](images/train_sample_006_recon.png) |

**Accuracy**: Character: 0.397, Color: 0.727

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 6](images/train_sample_006_ego_class_orig.png) | ![recon class 6](images/train_sample_006_ego_class_recon.png) |

**Class Accuracy**: 0.711

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (18 items):
------------------------------
  '#' (color  7)
  '(' (color  3)
  ')' (color  3)
  ')' (color  6)
  '*' (color  8)
  '+' (color  5)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '>' (color  7)
  '@' (color 15)
  '[' (color  3)
  '^' (color  5)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)
  '|' (color 15)

Reconstructed Bag (13 items):
------------------------------
  '#' (color  7)
  ')' (color  6)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 11 items
    '#' (color  7)
    ')' (color  6)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '>' (color  7)
    '@' (color 15)
    '{' (color 12)
    '|' (color  3)
    '|' (color  7)
  Missed items: 7 items
    '(' (color  3)
    ')' (color  3)
    '*' (color  8)
    '+' (color  5)
    '[' (color  3)
    '^' (color  5)
    '|' (color 15)
  False positives: 2 items
    '<' (color  7)
    '`' (color  7)

Performance Summary:
------------------------------
  Precision: 0.846 (11/13)
  Recall: 0.611 (11/18)
  F1-Score: 0.710
  Total unique items: 20
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 6](images/train_sample_006_pass_safe_orig.png) | ![recon pass safe 6](images/train_sample_006_pass_safe_recon.png) |

================================================================================

## Sample 8

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 7](images/train_sample_007_orig.png) | ![recon 7](images/train_sample_007_recon.png) |

**Accuracy**: Character: 0.281, Color: 0.884

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 7](images/train_sample_007_ego_class_orig.png) | ![recon class 7](images/train_sample_007_ego_class_recon.png) |

**Class Accuracy**: 0.760

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (17 items):
------------------------------
  '#' (color  7)
  '*' (color  1)
  '*' (color  7)
  '+' (color 10)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '[' (color  8)
  '`' (color  7)
  'd' (color 15)
  'r' (color 15)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (12 items):
------------------------------
  '#' (color  7)
  '*' (color  7)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 12 items
    '#' (color  7)
    '*' (color  7)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '>' (color  7)
    '@' (color 15)
    '`' (color  7)
    '|' (color  3)
    '|' (color  7)
  Missed items: 5 items
    '*' (color  1)
    '+' (color 10)
    '[' (color  8)
    'd' (color 15)
    'r' (color 15)
  False positives: 0 items

Performance Summary:
------------------------------
  Precision: 1.000 (12/12)
  Recall: 0.706 (12/17)
  F1-Score: 0.828
  Total unique items: 17
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 7](images/train_sample_007_pass_safe_orig.png) | ![recon pass safe 7](images/train_sample_007_pass_safe_recon.png) |

================================================================================

## Sample 9

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 8](images/train_sample_008_orig.png) | ![recon 8](images/train_sample_008_recon.png) |

**Accuracy**: Character: 0.198, Color: 0.694

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 8](images/train_sample_008_ego_class_orig.png) | ![recon class 8](images/train_sample_008_ego_class_recon.png) |

**Class Accuracy**: 0.661

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (14 items):
------------------------------
  '#' (color  7)
  ')' (color  6)
  '+' (color  3)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '@' (color 15)
  'Z' (color  3)
  '^' (color  8)
  '`' (color  7)
  'o' (color  7)
  '|' (color  7)

Reconstructed Bag (16 items):
------------------------------
  '#' (color  7)
  '(' (color  3)
  ')' (color  6)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '[' (color  3)
  '[' (color  6)
  '`' (color  7)
  '{' (color 12)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 10 items
    '#' (color  7)
    ')' (color  6)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '@' (color 15)
    '`' (color  7)
    '|' (color  7)
  Missed items: 4 items
    '+' (color  3)
    'Z' (color  3)
    '^' (color  8)
    'o' (color  7)
  False positives: 6 items
    '(' (color  3)
    '>' (color  7)
    '[' (color  3)
    '[' (color  6)
    '{' (color 12)
    '|' (color  3)

Performance Summary:
------------------------------
  Precision: 0.625 (10/16)
  Recall: 0.714 (10/14)
  F1-Score: 0.667
  Total unique items: 20
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 8](images/train_sample_008_pass_safe_orig.png) | ![recon pass safe 8](images/train_sample_008_pass_safe_recon.png) |

================================================================================

## Sample 10

### Ego Map Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig 9](images/train_sample_009_orig.png) | ![recon 9](images/train_sample_009_recon.png) |

**Accuracy**: Character: 0.248, Color: 0.289

### Ego Class Reconstruction

| Original | Reconstruction |
|---|---|
| ![orig class 9](images/train_sample_009_ego_class_orig.png) | ![recon class 9](images/train_sample_009_ego_class_recon.png) |

**Class Accuracy**: 0.446

### Bag Reconstruction

```
Bag Analysis:
========================================

Original Bag (18 items):
------------------------------
  '#' (color  7)
  '(' (color  3)
  ')' (color  6)
  '*' (color  1)
  '+' (color  2)
  '+' (color  5)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '[' (color  3)
  '^' (color 12)
  '`' (color  7)
  '|' (color  3)
  '|' (color  7)

Reconstructed Bag (15 items):
------------------------------
  '#' (color  7)
  '$' (color 11)
  ')' (color  6)
  '+' (color  1)
  '-' (color  3)
  '-' (color  7)
  '.' (color  7)
  '.' (color  8)
  '<' (color  7)
  '>' (color  7)
  '@' (color 15)
  '`' (color  7)
  'u' (color  3)
  '|' (color  3)
  '|' (color  7)

Accuracy Metrics:
------------------------------
  Correctly predicted: 12 items
    '#' (color  7)
    ')' (color  6)
    '-' (color  3)
    '-' (color  7)
    '.' (color  7)
    '.' (color  8)
    '<' (color  7)
    '>' (color  7)
    '@' (color 15)
    '`' (color  7)
    '|' (color  3)
    '|' (color  7)
  Missed items: 6 items
    '(' (color  3)
    '*' (color  1)
    '+' (color  2)
    '+' (color  5)
    '[' (color  3)
    '^' (color 12)
  False positives: 3 items
    '$' (color 11)
    '+' (color  1)
    'u' (color  3)

Performance Summary:
------------------------------
  Precision: 0.800 (12/15)
  Recall: 0.667 (12/18)
  F1-Score: 0.727
  Total unique items: 21
```

### Passability & Safety

| Original | Reconstruction |
|---|---|
| ![orig pass safe 9](images/train_sample_009_pass_safe_orig.png) | ![recon pass safe 9](images/train_sample_009_pass_safe_recon.png) |

## Overall Statistics

- **Average Character Accuracy**: 0.282
- **Average Color Accuracy**: 0.621
- **Total Samples**: 10
