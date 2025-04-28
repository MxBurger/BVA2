# OCR from scratch

## Table of Contents
- [Task 1](#task-1)
  - [Implementation](#implementation)
  - [Testing](#testing)
  - [Limitations](#limitations)
- [Task 2](#task-2)
  - [Implementation](#implementation-1)
  - [Testing](#testing-1)
  - [Limitations](#limitations-1)
- [Further considerations](#further-considerations)

## Task 1

### Solution idea

The basic idea as well as some code snippets was given during the lecture.

To implement an own OCR you generally have to do:
1. Binarize the input image
2. Separate text into lines and characters
3. Extract features from characters
4. Compare features to identify matching characters
5. Mark matching characters in the output image

### Implementation

The implemented features can be found in [ImageFeatures.py](./ImageFeatures.py).


### Testing
For testing purposes all possible characters and special characters were extracted from the given image. Some meta data information like the expected count and the position were also provided so that working with the tests is integrated in the implemented OCR algorithm and its parameters.

The testing happens in [main.py](./main.py).

During the test run each result of a character analyzation is printed.

At the end we show a overlayed map with all found characters.

### Limitations
- Font dependency - optimized for Arial Black
- Sensitivity to character spacing
- Limited rotation tolerance
- Binary threshold sensitivity


## Task 2

### Implementation

#### Character Dimension Region Shrinking

Region shrinking is used to enhance the accuracy of character detection by adjusting the bounding box to closely match the edges of the actual character. This step eliminates unnecessary white space around characters.

For this a flag `--shrink_chars` as well as the function `limit_characters_vertically()` was introduced in [OCRanalysis.py](./OCRanalysis.py). If the flag is not set to true, we won't shrink the chars.

What is done:
1. The first non empty rows from top and buttom are searched.
2. Extract the starting y position.
3. Extract the height of the character.

After that the two values are returned and the [`SubImageRegion`](./SubImageRegion.py) class used to actually shrink the image.

```python
def limit_character_vertically(char_region, BG_val):
    """
    Find the top and bottom bounds of the character within the region.
    Returns start_y and height of the actual character.
    """
    height = char_region.height
    width = char_region.width

    # Find the first non-empty row from top
    start_y = 0
    while start_y < height:
        empty_row = True
        for x in range(width):
            if char_region.subImgArr[start_y][x] != BG_val:
                empty_row = False
                break
        if not empty_row:
            break
        start_y += 1

    # Find the first non-empty row from bottom
    end_y = height - 1
    while end_y >= start_y:
        empty_row = True
        for x in range(width):
            if char_region.subImgArr[end_y][x] != BG_val:
                empty_row = False
                break
        if not empty_row:
            break
        end_y -= 1

    # Calculate actual character height
    char_height = end_y - start_y + 1

    # Ensure minimum height
    if char_height < 2:
        char_height = 2
        if start_y + char_height > height:
            start_y = height - char_height

    return start_y, char_height
```

#### Further additional features

**Hole Count** helps us to identificate characters with a different number of holes. This can be seen when comparing e. g. the characters `B` and `P`.

**Vertical Symmetry** helps to distinguish characters that are vertically symmetrical from those that are not. For example, `A` shows strong vertical symmetry, while `F` does not.

**Horizontal Symmetry**  supports the differentiation between characters based on their symmetry along the horizontal axis. For instance, `H` has high horizontal symmetry, whereas `L` lacks it.

**Aspect Ratio** is possible after successfully applying the region shrinking only on the real edges of the characters. ([see here](#character-dimension-region-shrinking)) It helps to differentiate e. g. wide characters like `w` from not that wide characters like `v`.

**Pixel Density** measures how many foreground pixels are occupied relative to the total amount of pixels of the bounding box of the character. It can help differentiate between characters like `M`, which tends to be denser, and `i`, which covers much less area.

#### Normalization Process



### Testing
This is the same like desribed in [Task 1](#task-1).

### Limitations

- Classification process itself
- Font style variations impact accuracy
- connectedLetters Problem

### Further considerations

