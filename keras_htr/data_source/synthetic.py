from .base import Source
from PIL import Image, ImageDraw, ImageFont, ImageOps
from wordfreq import top_n_list


class SyntheticSource(Source):
    def __init__(self, num_examples=1000):
        self._num_examples = num_examples

    def __iter__(self):
        words = top_n_list(lang='en', n=self._num_examples)

        for w in words:
            image = self.create_image(w)
            yield image, w

    def create_image(self, text):
        initial_image = Image.new('L', size=(100, 80))
        d = ImageDraw.Draw(initial_image)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)

        text_width, text_height = ImageDraw.ImageDraw.textsize(d, text, font=fnt)
        padding = 10
        image_width = text_width + 2 * padding
        image_height = text_height + 2 * padding

        line_image = Image.new('L', size=(image_width, image_height))
        d = ImageDraw.Draw(line_image)
        d.text((padding, padding), text, font=fnt, fill=(255,))

        return ImageOps.invert(line_image)
