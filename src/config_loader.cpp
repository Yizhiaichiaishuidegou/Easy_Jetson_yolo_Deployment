#include "config_loader.h"
#include <iostream>

bool ConfigLoader::loadConfig(const std::string& config_path) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        // 加载模型配置
        if (config["model"]) {
            auto model_node = config["model"];
            model_config_.engine_path = model_node["engine_path"].as<std::string>();
            model_config_.confidence_threshold = model_node["confidence_threshold"].as<float>(0.1f);
            model_config_.nms_threshold = model_node["nms_threshold"].as<float>(0.5f);
            model_config_.input_width = model_node["input_width"].as<int>(640);
            model_config_.input_height = model_node["input_height"].as<int>(640);
            model_config_.num_box = model_node["num_box"].as<int>(8400);
            
            // 加载类别名称
            if (model_node["class_names"]) {
                model_config_.class_names.clear();
                for (const auto& class_name : model_node["class_names"]) {
                    model_config_.class_names.push_back(class_name.as<std::string>());
                }
            }
        }
        
        // 加载输入配置
        if (config["input"]) {
            auto input_node = config["input"];
            input_config_.type = input_node["type"].as<std::string>("camera");
            if (input_node["source"]) {
                input_config_.video_path = input_node["source"]["video_path"].as<std::string>("");
                input_config_.image_path = input_node["source"]["image_path"].as<std::string>("");
                input_config_.camera_id = input_node["source"]["camera_id"].as<int>(0);
                input_config_.rtsp_url = input_node["source"]["rtsp_url"].as<std::string>("");
                input_config_.rtmp_url = input_node["source"]["rtmp_url"].as<std::string>("");
            }
            input_config_.video_id = input_node["video_id"].as<std::string>("default_video");
                        
            if (input_node["decoder"]) {
                std::string decoder_str = input_node["decoder"].as<std::string>("opencv");
                std::cout << "DEBUG: decoder_str = '" << decoder_str << "'" << std::endl;
                if (decoder_str == "gstreamer_nvdec") {
                    input_config_.decoder = DecoderType::GSTREAMER_NVDEC;
                } else {
                    input_config_.decoder = DecoderType::OPENCV;
                }
            } else {
                std::cout << "DEBUG: decoder node not found, using default" << std::endl;
                input_config_.decoder = DecoderType::OPENCV;
            }
        }
        
        // 加载输出配置
        if (config["output"]) {
            auto output_node = config["output"];
            output_config_.save_video = output_node["save_video"].as<bool>(true);
            output_config_.output_path = output_node["output_path"].as<std::string>("output.mp4");
            output_config_.save_detections = output_node["save_detections"].as<bool>(false);
            output_config_.detections_path = output_node["detections_path"].as<std::string>("detections.txt");
            output_config_.fps_log = output_node["fps_log"].as<bool>(true);
            output_config_.fps_log_path = output_node["fps_log_path"].as<std::string>("fps_data.txt");
        }
        
        // 加载显示配置
        if (config["display"]) {
            auto display_node = config["display"];
            display_config_.show_fps = display_node["show_fps"].as<bool>(true);
            display_config_.imshow_name = display_node["imshow_name"].as<std::string>("YOLO Detection");
            display_config_.window_width = display_node["window_width"].as<int>(1280);
            display_config_.window_height = display_node["window_height"].as<int>(720);
            std::string quit_key_str = display_node["quit_key"].as<std::string>("q");
            display_config_.quit_key = quit_key_str.empty() ? 'q' : quit_key_str[0];
            display_config_.show_video = display_node["show_video"].as<bool>(true);
        }
        
        // 加载性能配置
        if (config["performance"]) {
            auto perf_node = config["performance"];
            performance_config_.async_mode = perf_node["async_mode"].as<bool>(true);
            performance_config_.queue_size = perf_node["queue_size"].as<int>(5);
            performance_config_.gpu_id = perf_node["gpu_id"].as<int>(0);
            performance_config_.max_fps = perf_node["max_fps"].as<int>(30);
            if(perf_node["cpu_ids"]){
                performance_config_.cpu_ids.clear();
                for(const auto& cpu_id : perf_node["cpu_ids"]){
                    performance_config_.cpu_ids.push_back(cpu_id.as<int>());
                }

            }
            // 预处理后端配置
            std::string backend_str = perf_node["preprocess_backend"].as<std::string>("cuda");
            if (backend_str == "vpi_cuda") {
                performance_config_.preprocess_backend = PreprocessBackend::VPI_CUDA;
            } else if (backend_str == "vpi_vic") {
                performance_config_.preprocess_backend = PreprocessBackend::VPI_VIC;
            } else {
                performance_config_.preprocess_backend = PreprocessBackend::CUDA;
            }
            
            // DLA 配置
            performance_config_.use_dla = perf_node["use_dla"].as<bool>(false);
            performance_config_.dla_core = perf_node["dla_core"].as<int>(0);
            if (performance_config_.dla_core < 0 || performance_config_.dla_core > 1) {
                std::cerr << "Warning: Invalid DLA core " << performance_config_.dla_core 
                          << ", using core 0" << std::endl;
                performance_config_.dla_core = 0;
            }
        }
        
        // 加载调试配置
        if (config["debug"]) {
            auto debug_node = config["debug"];
            debug_config_.log_level = debug_node["log_level"].as<std::string>("INFO");
            debug_config_.save_debug_images = debug_node["save_debug_images"].as<bool>(false);
            debug_config_.debug_image_path = debug_node["debug_image_path"].as<std::string>("./debug_images/");
        }
        
        std::cout << "配置文件加载成功: " << config_path << std::endl;
        return true;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "加载配置文件失败: " << e.what() << std::endl;
        return false;
    }
}