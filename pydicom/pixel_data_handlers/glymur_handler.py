# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Use the Glymur python package to decode pixel transfer syntaxes."""

import io
import logging
import tempfile
try:
    import numpy
    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    import glymur
    glymur.set_option('lib.num_threads', 32)
    HAVE_GLYMUR = True
except ImportError:
    HAVE_GLYMUR = False

import pydicom.encaps
from pydicom.pixel_data_handlers.util import dtype_corrected_for_endianness
import pydicom.uid


logger = logging.getLogger('pydicom')

GlymurSupportedTransferSyntaxes = [
    pydicom.uid.JPEG2000Lossless,
]
GlymurJPEG2000TransferSyntaxes = [
    pydicom.uid.JPEG2000Lossless,
]

HANDLER_NAME = 'Glymur'

DEPENDENCIES = {
    'numpy': ('http://www.numpy.org/', 'NumPy'),
    'glymur': ('https://github.com/quintusdias/glymur', 'Glymur'),
}


def is_available():
    """Return True if the handler has its dependencies met."""
    return HAVE_NP and HAVE_PIL


def supports_transfer_syntax(transfer_syntax):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.

        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return transfer_syntax in GlymurSupportedTransferSyntaxes


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    should_change = dicom_dataset.SamplesPerPixel == 3
    return False


def get_pixeldata(dicom_dataset):
    """Use Glymur to decompress compressed Pixel Data.

    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.

    Raises
    ------
    ImportError
        If PIL is not available.

    NotImplementedError
        if the transfer syntax is not supported

    TypeError
        if the pixel data type is unsupported
    """
    logger.debug("Trying to use Glymur to read pixel array "
                 "(has glymur = %s)", HAVE_GLYMUR)
    transfer_syntax = dicom_dataset.file_meta.TransferSyntaxUID
    if not HAVE_PIL:
        msg = ("The pillow package is required to use pixel_array for "
               "this transfer syntax {0}, and pillow could not be "
               "imported.".format(transfer_syntax.name))
        raise ImportError(msg)


    if transfer_syntax not in GlymurSupportedTransferSyntaxes:
        msg = ("this transfer syntax {0}, can not be read because "
               "Glymur does not support this syntax"
               .format(transfer_syntax.name))
        raise NotImplementedError(msg)

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_format = numpy.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    numpy_format = dtype_corrected_for_endianness(
        dicom_dataset.is_little_endian, numpy_format)

    # decompress here
    if transfer_syntax in GlymurJPEG2000TransferSyntaxes:
        logger.debug("This is a JPEG 2000 format")
        generic_jpeg_file_header = b''
        # generic_jpeg_file_header = b'\x00\x00\x00\x0C\x6A'
        #     b'\x50\x20\x20\x0D\x0A\x87\x0A'
        frame_start_from = 0
    else:
        raise Exception('Transfer syntax not understood {}'.format(transfer_syntax))
        
    try:
        UncompressedPixelData = bytearray()
        if ('NumberOfFrames' in dicom_dataset and
                dicom_dataset.NumberOfFrames > 1):
            # multiple compressed frames
            CompressedPixelDataSeq = \
                pydicom.encaps.decode_data_sequence(
                    dicom_dataset.PixelData)
            for frame in CompressedPixelDataSeq:
                data = generic_jpeg_file_header + \
                    frame[frame_start_from:]
                fio = io.BytesIO(data)
                try:
                    with tempfile.NamedTemporaryFile() as f:
                        f.write(fio)
                        f.seek(0)
                        decompressed_image = glymur.Jp2k(f.name)[:]
                except IOError as e:
                    raise NotImplementedError(e.strerror)
                UncompressedPixelData.extend(decompressed_image.tobytes())
        else:
            # single compressed frame
            pixel_data = pydicom.encaps.defragment_data(
                dicom_dataset.PixelData)
            pixel_data = generic_jpeg_file_header + \
                pixel_data[frame_start_from:]
            try:
                fio = io.BytesIO(pixel_data)
                try:
                    with tempfile.NamedTemporaryFile() as f:
                        f.write(fio)
                        f.seek(0)
                        decompressed_image = glymur.Jp2k(f.name)[:]
            except IOError as e:
                raise NotImplementedError(e.strerror)
            UncompressedPixelData.extend(decompressed_image.tobytes())
    except Exception:
        raise

    logger.debug(
        "Successfully read %s pixel bytes", len(UncompressedPixelData)
    )

    pixel_array = numpy.frombuffer(UncompressedPixelData, numpy_format)

    #if (transfer_syntax in
    #        PillowJPEG2000TransferSyntaxes and
    #        dicom_dataset.BitsStored == 16):
    #    # WHY IS THIS EVEN NECESSARY??
    #    pixel_array &= 0x7FFF

    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"

    return pixel_array
