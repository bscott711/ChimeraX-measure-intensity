"""Generates a QR code with a logo."""

from docopt import docopt
from qrcode import QRCode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import CircleModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask


def generate_qr(logo_path, qr_color, url, file_name):
    """Generate a QR code with a logo in the middle, and the pixels of specific color."""
    qr_code = QRCode(error_correction=ERROR_CORRECT_H)
    # adding URL or text to QR code
    qr_code.add_data(url)
    # generating QR code
    qr_code.make()
    # adding color and logo to QR code
    qr_img = qr_code.make_image(
        image_factory=StyledPilImage,
        module_drawer=CircleModuleDrawer(),
        color_mask=SolidFillColorMask(front_color=qr_color),
        embeded_image_path=logo_path,
    )
    qr_img.save(file_name)


if __name__ == "__main__":
    USAGE = """Usage: generate_qr.py [-h] [-c COLOR] [-l LOGO_PATH] URL FILE_NAME

    Generates a QR code with a logo.

    Arguments:
    URL             The URL or text to encode in the QR code.
    FILE_NAME       The name of the output file to save the QR code to.

    Options:
    -h --help       Show this help message and exit.
    -c COLOR        The color of the QR code. Specify as three comma-separated values representing RGB color channels. [default: 38,74,113]
    -l LOGO_PATH    The path to the logo image file to embed in the QR code. [default: LabLogo.png]
    """
    args = docopt(USAGE)

    # Convert color argument from string to tuple
    color = tuple(map(int, args['-c'].split(',')))

    # Call generate_qr function with arguments
    generate_qr(args['-l'], color, args['URL'], args['FILE_NAME'])
