"""Generates a QRcode."""
from qrcode import QRCode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import CircleModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask


def generate_qr(
    logo_path="LabLogo.png",
    qr_color=(38, 74, 113),
    url="http://babylonjs-viewer.glitch.me",
    file_name="lablogoQR.png",
):
    """Generate a qrcode to a new URL with a logo in the middle, and the pixels of specific color"""
    qr_code = QRCode(error_correction=ERROR_CORRECT_H)
    # adding URL or text to QRcode
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
    generate_qr()
