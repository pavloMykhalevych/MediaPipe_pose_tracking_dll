// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_UTIL_POSE_PRESENT_CALCULATOR_H
#define MEDIAPIPE_CALCULATORS_UTIL_POSE_PRESENT_CALCULATOR_H

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {

// A calculator that counts the size of the input vector. It was created to
// aid in polling packets in the output stream synchronously. If there is
// a clock stream, it will output a value of 0 even if the input vector stream
// is empty. If not, it will output some value only if there is an input vector.
// The clock stream must have the same time stamp as the vector stream, and
// it must be the stream where packets are transmitted while the graph is
// running. (e.g. Any input stream of graph)
//
// It is designed to be used like:
//
// Example config:
// node {
//   calculator: "CountingWithVectorSizeCalculator"
//   input_stream: "CLOCK:triger_signal"
//   input_stream: "VECTOR:input_vector"
//   output_stream: "COUNT:vector_count"
// }
//
// node {
//   calculator: "CountingWithVectorSizeCalculator"
//   input_stream: "VECTOR:input_vector"
//   output_stream: "COUNT:vector_count"
// }

template <typename VectorT>
class PosePresentCalculator : public CalculatorBase {
public:
  static ::mediapipe::Status GetContract(CalculatorContract *cc) {
    if (cc->Inputs().HasTag("CLOCK")) {
      cc->Inputs().Tag("CLOCK").SetAny();
    }

    RET_CHECK(cc->Inputs().HasTag("LANDMARKS"));
    cc->Inputs().Tag("LANDMARKS").Set<VectorT>();
    RET_CHECK(cc->Outputs().HasTag("BOOL"));
    cc->Outputs().Tag("BOOL").Set<bool>();

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext *cc) {
    std::unique_ptr<bool> pose_count;
    if (!cc->Inputs().Tag("LANDMARKS").IsEmpty()) {
      const auto &landmarks = cc->Inputs().Tag("LANDMARKS").Get<VectorT>();
      pose_count = absl::make_unique<bool>(landmarks.landmark_size());
    } else {
      pose_count = absl::make_unique<bool>(false);
    }
    cc->Outputs().Tag("BOOL").Add(pose_count.release(), cc->InputTimestamp());
    
    return ::mediapipe::OkStatus();
  };
};

} // namespace mediapipe

#endif // MEDIAPIPE_CALCULATORS_UTIL_POSE_PRESENT_CALCULATOR_H