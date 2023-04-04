import fitz
import qrcode

# obtener la información del PDF
def get_pdf_info(file_path, start_page=None, end_page=None):
    # abrir el archivo PDF
    with fitz.open(file_path) as pdf:
        # iterar por cada página del PDF
        for page in pdf:
            # si se especificó un rango de páginas, saltar las páginas que no están en el rango
            if start_page is not None and page.number + 1 < start_page:
                continue
            if end_page is not None and page.number + 1 > end_page:
                break
            # extraer el texto de la página
            text = page.get_text()
            # agregar el texto al resultado
            yield text

# generar el código QR
def generate_qr_code(data):
    # crear el objeto QRCode
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    # agregar los datos al objeto QRCode
    qr.add_data(data)
    # crear la imagen del código QR
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# obtener la información del PDF y generar el código QR
def generate_qr_code_from_pdf(file_path, start_page=None, end_page=None):
    # obtener la información del PDF
    pdf_info = "\n".join(get_pdf_info(file_path, start_page=start_page, end_page=end_page))
    # generar el código QR a partir de la información del PDF
    qr_code = generate_qr_code(pdf_info)
    return qr_code

# ejemplo de uso
qr_code = generate_qr_code_from_pdf("archivo.pdf", start_page=1, end_page=3)
qr_code.save("qr_code.png")