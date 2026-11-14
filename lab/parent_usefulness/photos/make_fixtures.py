"""Draw original, public synthetic worksheet inputs. No family data or model output."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)
FONT = ImageFont.truetype("DejaVuSans.ttf", 30)
SMALL = ImageFont.truetype("DejaVuSans.ttf", 24)


def page(title, instruction):
    image = Image.new("RGB", (1000, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 40), title, fill="black", font=FONT)
    draw.text((45, 110), instruction, fill="black", font=SMALL)
    return image, draw


def save(image, name):
    image.save(OUT / f"{name}.png")


image, draw = page("1. Longer and shorter", "Circle the longer strip in each pair.")
for y, widths in ((220, (520, 270)), (490, (230, 490))):
    for index, width in enumerate(widths):
        draw.rectangle((100, y + index * 85, 100 + width, y + index * 85 + 42), fill="#4b91be")
save(image, "length")

image, draw = page("2. Above and below", "Point to the ball above the box. Draw one below.")
draw.rectangle((350, 350, 650, 520), outline="black", width=5)
draw.ellipse((450, 190, 550, 290), fill="#d88d36", outline="black", width=3)
draw.line((350, 700, 650, 700), fill="#bbbbbb", width=3)
save(image, "position")

image, draw = page("3. Empty and full", "Circle the full cup. Point to the empty cup.")
for x in (170, 580):
    draw.rectangle((x, 270, x + 220, 610), outline="black", width=5)
draw.rectangle((584, 290, 796, 606), fill="#75c4e8")
save(image, "capacity")

image, draw = page("4. Repeating patterns", "Continue each pattern. What repeats?")
for row, unit in enumerate((("#d34141", "#315bab"), ("#d34141", "#d34141", "#315bab"))):
    for i in range(8):
        x, y = 70 + i * 110, 270 + row * 240
        if i < 6:
            draw.ellipse((x, y, x + 65, y + 65), fill=unit[i % len(unit)], outline="black", width=2)
        else:
            draw.line((x, y + 65, x + 65, y + 65), fill="black", width=3)
save(image, "patterns")

image, draw = page("5. Equal groups", "Which two plates have the same number of dots?")
for x, count in ((60, 3), (375, 5), (690, 3)):
    draw.ellipse((x, 290, x + 245, 560), outline="black", width=4)
    for i in range(count):
        dx, dy = x + 50 + (i % 3) * 55, 370 + (i // 3) * 70
        draw.ellipse((dx, dy, dx + 30, dy + 30), fill="#496f56")
save(image, "groups")

image, draw = page("Beginning sounds: /m/", "Say each word. Circle the words that START with /m/.")
for i, word in enumerate(("moon", "sun", "mop", "jam")):
    x, y = 70 + (i % 2) * 450, 240 + (i // 2) * 240
    draw.rounded_rectangle((x, y, x + 350, y + 180), radius=15, outline="black", width=3)
    draw.text((x + 80, y + 60), word, fill="black", font=FONT)
save(image, "sounds")

image, draw = page("Equal parts: halves", "Circle the squares split into two equal parts.")
for i in range(3):
    x, y = 65 + i * 315, 300
    draw.rectangle((x, y, x + 250, y + 250), outline="black", width=5)
    if i == 0:
        draw.line((x + 125, y, x + 125, y + 250), fill="black", width=4)
    elif i == 1:
        draw.line((x + 65, y, x + 65, y + 250), fill="black", width=4)
    else:
        draw.line((x, y, x + 250, y + 250), fill="black", width=4)
draw.text((60, 640), "Two pieces are halves only if they are equal shares.", fill="black", font=SMALL)
save(image, "halves")
