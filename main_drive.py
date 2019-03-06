import sys
import time

import gi

from drive import Drive
from motor.client import set_target_degree

gi.require_version("Tcam", "0.1")
gi.require_version("Gst", "1.0")

from gi.repository import Tcam, Gst, GLib

import numpy as np

from multiprocessing import Process

display_buffers = []

drive = Drive()


def ask_torch_and_set_motor(img_array):
    result = drive.forward(img_array)
    set_target_degree(result)


def show_img(display_input, img, display_buffers):
    bytebuffer = img.tobytes()
    display_buffers.append(bytebuffer)
    new_buf = Gst.Buffer.new_wrapped_full(Gst.MemoryFlags.READONLY, bytebuffer, len(bytebuffer), 0, None,
                                          lambda x: display_buffers.pop(0))
    display_input.emit("push-buffer", new_buf)


counter = 0


def callback(sink, display_input, display_buffers, index):
    global counter
    """
    This function will be called in a separate thread when our appsink
    says there is data for us."""
    sample = sink.emit("pull-sample")
    start_time = time.time()
    counter += 1
    if sample:
        buf = sample.get_buffer()

        caps = sample.get_caps()
        width = caps[0].get_value("width")
        height = caps[0].get_value("height")

        try:
            res, mapinfo = buf.map(Gst.MapFlags.READ)
            # actual image buffer and size
            # data = mapinfo.data
            # size = mapinfo.size

            # Create a numpy array from the data
            img_array = np.asarray(bytearray(mapinfo.data), dtype=np.uint8)
            ask_torch_and_set_motor(img_array)
            # Give the array the correct dimensions of the video image
            img = img_array.reshape((height, width, 4))
            counter += 1
            # file_name = "images/%s_%s_%s.jpg" % (time.strftime("%Y-%m-%d"), index, counter)
            # size = (width, height)
            # im = Image.frombytes("RGBX", size, img, "raw")
            exif_dict = {}

            if counter % (index + 1) == 0:
                show_img(display_input, img, display_buffers)
        # except Exception as e:
        #     print("index:%s" % index, e)
        finally:
            buf.unmap(mapinfo)
    return Gst.FlowReturn.OK


def process_method(source, fmt, TARGET_FORMAT, index):
    if fmt.get_name() == "video/x-bayer":
        fmt.set_name("video/x-raw")
        fmt.set_value("format", "BGRx")
    # Use a capsfilter to determine the video format of the camera source
    capsfilter = Gst.ElementFactory.make("capsfilter")
    capsfilter.set_property("caps", Gst.Caps.from_string(fmt.to_string()))
    # Add a small queue. Everything behind this queue will run in a separate
    # thread.
    queue = Gst.ElementFactory.make("queue")
    queue.set_property("leaky", True)
    queue.set_property("max-size-buffers", 2)
    # Add a videoconvert and a videoscale element to convert the format of the
    # camera to the target format for opencv
    convert = Gst.ElementFactory.make("videoconvert")
    scale = Gst.ElementFactory.make("videoscale")
    # Add an appsink. This element will receive the converted video buffers and
    # pass them to opencv
    output = Gst.ElementFactory.make("appsink")
    output.set_property("caps", Gst.Caps.from_string(TARGET_FORMAT))
    output.set_property("emit-signals", True)
    pipeline = Gst.Pipeline.new()

    # Add all elements
    pipeline.add(source)
    pipeline.add(capsfilter)
    pipeline.add(queue)
    pipeline.add(convert)
    pipeline.add(scale)
    pipeline.add(output)

    # Link the elements
    source.link(capsfilter)
    capsfilter.link(queue)
    queue.link(convert)
    convert.link(scale)
    scale.link(output)

    # Usually one would use cv2.imgshow(...) to display an image but this is
    # tends to hang in threaded environments. So we create a small display
    # pipeline which we could use to display the opencv buffers.
    display_pipeline = Gst.parse_launch("appsrc name=src%s ! videoconvert ! ximagesink" % index)
    display_input = display_pipeline.get_by_name("src%s" % index)
    display_input.set_property("caps", Gst.Caps.from_string(TARGET_FORMAT))
    output.connect("new-sample", callback, display_input, display_buffers, index)
    display_pipeline.set_state(Gst.State.PLAYING)

    pipeline.set_state(Gst.State.PLAYING)
    print("Press Ctrl-C to stop")
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Ctrl-C pressed, terminating")

    # this stops the pipeline and frees all resources
    pipeline.set_state(Gst.State.NULL)
    display_pipeline.set_state(Gst.State.NULL)
    source.set_state(Gst.State.NULL)
    print('all cleared')


def main():
    Gst.init(sys.argv)  # init gstreamer

    # We create a source element to retrieve a device list through it
    source = Gst.ElementFactory.make("tcambin")
    serials = source.get_device_serials()
    sources = []
    format1 = Gst.Structure.from_string(
        "video/x-raw, format=(string)YUY2, width=(int)1920, height=(int)1080, framerate=(fraction)30/1;")
    format1 = format1[0]
    format2 = Gst.Structure.from_string(
        "video/x-bayer, format=(string)rggb, width=(int)2592, height=(int)1944, framerate=(fraction)15/1;")
    format2 = format2[0]
    TARGET_FORMAT1 = "video/x-raw,width=1920,height=1080,format=RGBx"
    TARGET_FORMAT2 = "video/x-raw,width=2592,height=1944,format=RGBx"
    # processes = []
    if serials:
        index = 0
        for serial in serials:
            index += 1
            format = format1
            TARGET_FORMAT = TARGET_FORMAT1
            if serial == '1810541':
                format = format2
                TARGET_FORMAT = TARGET_FORMAT2

            # source = Gst.ElementFactory.make("tcambin")
            source.set_property("serial", serial)
            # sources.append(source)
            # p = Process(target=process_method, args=(source, format, TARGET_FORMAT, index))
            # processes.append(p)
            # p.start()
            process_method(source, format, TARGET_FORMAT, index)
        # for p in processes:
        #     p.join()



if __name__ == '__main__':
    main()
