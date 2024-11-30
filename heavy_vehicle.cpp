#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "nvbufsurftransform.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
 
#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
  } \
} while (0)

using namespace cv;
using namespace std;
#define MAX_DISPLAY_LEN 64
#define PGIE_CLASS_ID_BUS 5
#define PGIE_CLASS_ID_TRUCK 7
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 780
#define MUXER_BATCH_TIMEOUT_USEC 40000
static gboolean PERF_MODE = FALSE;

Mat ones(MUXER_OUTPUT_HEIGHT,MUXER_OUTPUT_WIDTH,CV_16UC1,Scalar(5));

#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

gint frame_number = 0;

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* pad created by the decodebin is for video and not audio */
  if (!strncmp (name, "video", 0)) {
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object, gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strrstr (name, "source") == name) {
        g_object_set(G_OBJECT(object),"drop-on-latency",true,NULL);
  }

}

GstElement * create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };
  g_snprintf (bin_name, 15, "source-bin-%02d", index);

  /* Create a source GstBin to abstract this bin's content from the rest of the pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.*/
  if (PERF_MODE) {
    uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
    g_object_set (G_OBJECT (uri_decode_bin), "file-loop", TRUE, NULL);
    g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
  } else {
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
  }

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);
  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /*create a ghost pad for the source bin which will act as a proxy for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the cb_newpad callback, we will set the ghost pad target to the video decoder src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}


static GstPadProbeReturn infer_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
 {
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  NvDsMetaList *l_frame = NULL;
  char file_name[128];
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList * l_obj = NULL;  
  GstMapInfo in_map_info;

  if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) 
  {
    g_print("Error: Failed to map gst buffer\n");
    return GST_PAD_PROBE_OK;
  }
  NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
  // TODO for cuda device memory we need to use cudamemcpy
  NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

  int offset = 0;
  for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;l_obj = l_obj->next) 
  {   
    obj_meta = (NvDsObjectMeta *) (l_obj->data);
    int lower_mid_x = (int)(obj_meta->rect_params.left + obj_meta->rect_params.width/2);
    int lower_mid_y = (int)(obj_meta->rect_params.top + obj_meta->rect_params.height);
  }

    guint height = surface->surfaceList[frame_meta->batch_id].height;
    guint width = surface->surfaceList[frame_meta->batch_id].width;
    NvBufSurface *inter_buf = nullptr;
    NvBufSurfaceCreateParams create_params;
    create_params.gpuId = surface->gpuId;
    create_params.width = width;
    create_params.height = height;
    create_params.size = 0;
    create_params.colorFormat = NVBUF_COLOR_FORMAT_BGRA;
    create_params.layout = NVBUF_LAYOUT_PITCH;

  #ifdef __aarch64__
      create_params.memType = NVBUF_MEM_DEFAULT;
  #else
      create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
  #endif

  // Create another scratch RGBA NvBufSurface
  if (NvBufSurfaceCreate(&inter_buf, 1, &create_params) != 0) 
  {
    GST_ERROR("Error: Could not allocate internal buffer ");
    return GST_PAD_PROBE_OK;
  }

    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    NvBufSurfTransformRect src_rect;
    NvBufSurfTransformRect dst_rect;
    cudaStream_t cuda_stream;
    CHECK_CUDA_STATUS(cudaStreamCreate(&cuda_stream),"Could not create cuda stream");
    transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
    transform_config_params.gpu_id = surface->gpuId;
    transform_config_params.cuda_stream = cuda_stream;
    NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams(&transform_config_params);
    if (err != NvBufSurfTransformError_Success) {
      cout << "NvBufSurfTransformSetSessionParams failed with error " << err
           << endl;
      return GST_PAD_PROBE_OK;
    }
    /* Set the transform ROIs for source and destination, only do the color format conversion*/
    src_rect = {0, 0, width, height};
    dst_rect = {0, 0, width, height};

    /* Set the transform parameters */
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dst_rect;
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform_params.transform_flip = NvBufSurfTransform_None;
    transform_params.transform_filter = NvBufSurfTransformInter_Algo3;

    /* Transformation format conversion */
    err = NvBufSurfTransform(surface, inter_buf, &transform_params);
    if (err != NvBufSurfTransformError_Success) {
      cout << "NvBufSurfTransform failed with error %d while converting buffer"
           << err << endl;
      return GST_PAD_PROBE_OK;
    }

    if (NvBufSurfaceMap(inter_buf, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
      cout << "map error" << endl;
      break;
    }

    NvBufSurfaceUnMap(inter_buf, 0, -1);
    NvBufSurfaceDestroy(inter_buf);
  }

  NvBufSurfaceUnMap(surface, -1, -1);
  gst_buffer_unmap(buf, &in_map_info);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad and updates params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    guint two_wheeler_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        int offset1 = 0;

        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (char*)g_malloc0 (MAX_DISPLAY_LEN);

        
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_BUS || obj_meta->class_id == PGIE_CLASS_ID_TRUCK) {
              vehicle_count++;
                offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "ALERT!   HEAVY VEHICLES ARE NOT ALLOWED IN THIS ZONE....!!!!!!");
                
            }
        }
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    // g_print ("HEAVY VEHICLE DETETCTED");
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL;

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL,*infer_sink_pad = NULL;

  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  if (argc != 2) {
    g_printerr ("Usage: %s file:///<absolute-path-to-yml file>\n", argv[0]);
    g_printerr ("OR: %s file:///<absolute-path-to-H264 filename>\n", argv[0]);
    return -1;
  }
  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  }

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest1-pipeline");
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  GstPad *sinkpad, *srcpad;
  gchar pad_name[16] = { };
  GstElement *source_bin= NULL;
  source_bin = create_source_bin (0, argv[0 + 1]);

  if (!source_bin) {
    g_printerr ("Failed to create source bin. Exiting.\n");
    return -1;
  }
  srcpad = gst_element_get_static_pad (source_bin, "src");

  gst_bin_add (GST_BIN (pipeline), source_bin);

  g_snprintf (pad_name, 15, "sink_%u", 1);
  sinkpad = gst_element_get_request_pad (streammux, pad_name);


  if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

  /* Create nvstreammux instance to form batches from one or more sources. */
 
  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer or nvinferserver to run inferencing on decoder's output, behaviour of inferencing is set through config file */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }
  
  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  if(prop.integrated) {
    sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
  } else {
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  }

  if (!nvvidconv || !sink || !nvosd) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  // g_object_set (
  //   G_OBJECT (pgie),
  //   "config-file-path", 
  //   "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test1/dstest1_pgie_config.txt", 
  //   NULL);
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "vehicle_pgie_config.txt", NULL);
  
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline and add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),streammux, pgie,nvosd,nvvidconv, sink, NULL);


  if (!gst_element_link_many (streammux, pgie,nvvidconv, nvosd,sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
  g_print ("Added elements to bin\n");

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) 
  {
    g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
    g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (srcpad);
  gst_object_unref (sinkpad);

  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  infer_sink_pad = gst_element_get_static_pad (nvvidconv, "sink");
  if (!infer_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (infer_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        infer_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (infer_sink_pad);


  /* Set the pipeline to "playing" state */
  g_print ("Using file: %s\n", argv[1]);
  // GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);
  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}






























