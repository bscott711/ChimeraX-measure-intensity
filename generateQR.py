from qrcode import QRCode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import CircleModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask


def generate_qr(logo_path='LabLogo.png', QRcolor=(38, 74, 113), url='http://babylonjs-viewer.glitch.me', file_name='lablogoQR.png'):
    """Generate a qrcode to a new URL with a logo in the middle, and the pixels of specific color"""
    qr = QRCode(
        error_correction=ERROR_CORRECT_H
    )
    # adding URL or text to QRcode
    qr.add_data(url)
    # generating QR code
    qr.make()
    # adding color and logo to QR code
    QRimg = qr.make_image(
        image_factory=StyledPilImage, module_drawer=CircleModuleDrawer(),
        color_mask=SolidFillColorMask(front_color=QRcolor), embeded_image_path=logo_path)
    QRimg.save(file_name)


if __name__ == '__main__':
    generate_qr()
