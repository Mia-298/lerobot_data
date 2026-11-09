#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Camera.h"

namespace py = pybind11;
using namespace gen3d;

namespace {

class CameraError : public std::runtime_error {
public:
    CameraError(const std::string& operation, int code)
        : std::runtime_error(operation + " failed with CamSDK error " + std::to_string(code)),
          code(code) {}

    int code;
};

void check_result(const std::string& operation, int code) {
    if (code != AC_OK) {
        throw CameraError(operation, code);
    }
}

py::dict camera_info_to_dict(const CameraInfo& info) {
    py::dict result;
    result["name"] = info.cameraName;
    result["ip"] = info.cameraIP;
    result["user_ip"] = info.userIP;
    result["port"] = info.port;
    result["serial_number"] = info.serialNum;
    result["mac_address"] = info.macAddr;
    result["model"] = info.cameraModel;
    result["camera_version"] = info.cameraSystemVersion;
    result["sdk_version"] = info.sdkVersion;
    result["is_dhcp"] = info.isDHCP;
    result["error_code"] = info.errorCode;
    result["rgb_status"] = info.rgbStatus;
    return result;
}

class CameraWrapper {
public:
    CameraWrapper() : camera_(CreateCamera(CamType::CamPro)) {
        if (camera_ == nullptr) {
            throw std::runtime_error("CreateCamera returned null");
        }
    }

    ~CameraWrapper() {
        close_noexcept();
        if (camera_ != nullptr) {
            DestoryCamera(camera_);
            camera_ = nullptr;
        }
    }

    CameraWrapper(const CameraWrapper&) = delete;
    CameraWrapper& operator=(const CameraWrapper&) = delete;

    std::vector<py::dict> discover(uint32_t timeout_ms = 1000) {
        std::vector<CameraInfo> cameras;
        int result;
        {
            py::gil_scoped_release release;
            result = camera_->DiscoverCameras(cameras, timeout_ms);
        }
        check_result("DiscoverCameras", result);

        std::vector<py::dict> output;
        output.reserve(cameras.size());
        for (const auto& camera : cameras) {
            output.push_back(camera_info_to_dict(camera));
        }
        return output;
    }

    py::dict open(
        const std::string& camera_ip = "",
        uint32_t discover_timeout_ms = 1000,
        uint32_t open_timeout_ms = 60000
    ) {
        close_noexcept();

        std::vector<CameraInfo> cameras;
        int result;
        {
            py::gil_scoped_release release;
            result = camera_->DiscoverCameras(cameras, discover_timeout_ms);
        }
        check_result("DiscoverCameras", result);
        if (cameras.empty()) {
            throw CameraError("DiscoverCameras", AC_E_NO_CAMERA);
        }

        auto selected = cameras.begin();
        if (!camera_ip.empty()) {
            selected = std::find_if(
                cameras.begin(), cameras.end(),
                [&camera_ip](const CameraInfo& info) { return info.cameraIP == camera_ip; }
            );
            if (selected == cameras.end()) {
                throw std::runtime_error("camera " + camera_ip + " was not discovered");
            }
        }
        camera_info_ = *selected;

        {
            py::gil_scoped_release release;
            result = camera_->Open(camera_info_, open_timeout_ms);
        }
        check_result("Open", result);
        opened_ = true;

        camera_info_.outputSettings.sendPoint3D = false;
        camera_info_.outputSettings.sendPointUV = false;
        camera_info_.outputSettings.sendTriangleIndices = false;
        camera_info_.outputSettings.sendDepthmap = true;
        camera_info_.outputSettings.sendNormals = false;
        camera_info_.outputSettings.sendPointColor = false;
        camera_info_.outputSettings.sendTexture = camera_info_.rgbStatus == AC_OK;
        camera_info_.outputSettings.sendRemapTexture = false;

        result = camera_->SetValue(
            camera_info_, Capture_WorkMode, static_cast<int32_t>(Camera_SoftTrigger)
        );
        if (result != AC_OK) {
            close_noexcept();
            check_result("SetValue(Capture_WorkMode)", result);
        }
        return camera_info_to_dict(camera_info_);
    }

    py::dict capture(bool convert_bgr_to_rgb = true) {
        require_open();

        FrameData frame;
        int result;
        {
            py::gil_scoped_release release;
            result = camera_->Capture(camera_info_, frame);
        }
        check_result("Capture", result);

        py::dict output;
        output["frame_index"] = frame.frameInfo.frameIndex;
        output["timestamp"] = frame.frameInfo.frameTimestamp;

        if (frame.texture != nullptr && frame.textureSize > 0) {
            const py::ssize_t height = camera_info_.camParam.textureHeight;
            const py::ssize_t width = camera_info_.camParam.textureWidth;
            const size_t expected = static_cast<size_t>(height * width * 3);
            if (frame.textureSize < expected) {
                throw std::runtime_error(
                    "texture buffer is smaller than textureWidth * textureHeight * 3"
                );
            }

            py::array_t<uint8_t> rgb({height, width, py::ssize_t{3}});
            auto* destination = rgb.mutable_data();
            if (convert_bgr_to_rgb) {
                for (size_t pixel = 0; pixel < expected; pixel += 3) {
                    destination[pixel] = frame.texture[pixel + 2];
                    destination[pixel + 1] = frame.texture[pixel + 1];
                    destination[pixel + 2] = frame.texture[pixel];
                }
            } else {
                std::memcpy(destination, frame.texture, expected);
            }
            output["rgb"] = std::move(rgb);
        } else {
            output["rgb"] = py::none();
        }

        if (frame.depthmap != nullptr && frame.depthmapSize > 0) {
            const bool aligned_to_rgb = camera_info_.camParam.depthType == DEPTH_MAP_ALIGN_RGB;
            const py::ssize_t height = aligned_to_rgb
                ? camera_info_.camParam.textureHeight
                : camera_info_.camParam.irHeight;
            const py::ssize_t width = aligned_to_rgb
                ? camera_info_.camParam.textureWidth
                : camera_info_.camParam.irWidth;
            const size_t expected = static_cast<size_t>(height * width);
            if (frame.depthmapSize < expected) {
                throw std::runtime_error(
                    "depth buffer is smaller than the dimensions reported by CameraParam"
                );
            }

            py::array_t<float> depth({height, width});
            std::memcpy(depth.mutable_data(), frame.depthmap, expected * sizeof(float));
            output["depth"] = std::move(depth);
        } else {
            output["depth"] = py::none();
        }

        return output;
    }

    void close() {
        close_noexcept();
    }

    bool is_open() const {
        return opened_;
    }

    py::dict info() const {
        require_open();
        return camera_info_to_dict(camera_info_);
    }

private:
    void require_open() const {
        if (!opened_) {
            throw std::runtime_error("camera is not open");
        }
    }

    void close_noexcept() noexcept {
        if (camera_ != nullptr && opened_) {
            try {
                camera_->Close(camera_info_);
            } catch (...) {
            }
            opened_ = false;
        }
    }

    Camera* camera_ = nullptr;
    CameraInfo camera_info_;
    bool opened_ = false;
};

}  // namespace

PYBIND11_MODULE(_camsdk, module) {
    module.doc() = "Python bindings for the CamSDK CameraPro SDK";

    py::register_exception<CameraError>(module, "CameraError");

    py::class_<CameraWrapper>(module, "Camera")
        .def(py::init<>())
        .def("discover", &CameraWrapper::discover, py::arg("timeout_ms") = 1000)
        .def(
            "open", &CameraWrapper::open,
            py::arg("camera_ip") = "",
            py::arg("discover_timeout_ms") = 1000,
            py::arg("open_timeout_ms") = 60000
        )
        .def(
            "capture", &CameraWrapper::capture,
            py::arg("convert_bgr_to_rgb") = true
        )
        .def("close", &CameraWrapper::close)
        .def_property_readonly("is_open", &CameraWrapper::is_open)
        .def_property_readonly("info", &CameraWrapper::info)
        .def("__enter__", [](CameraWrapper& self) -> CameraWrapper& { return self; })
        .def("__exit__", [](CameraWrapper& self, py::object, py::object, py::object) {
            self.close();
        });
}

