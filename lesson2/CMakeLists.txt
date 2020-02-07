
list(APPEND udacity_sources
    ${CMAKE_CURRENT_LIST_DIR}/tere.cu
    )

add_library(udacity-obj
            OBJECT
            ${udacity_sources}
            )

add_library(udacity-static
            STATIC
            $<TARGET_OBJECTS:udacity-obj>
            )

set_target_properties(udacity-static
            PROPERTIES
            OUTPUT_NAME udacity
            CUDA_SEPARABLE_COMPILATION ON
            )

target_include_directories(udacity-obj PRIVATE ${CMAKE_CURRENT_LIST_DIR}/..)
target_compile_features(udacity-obj PUBLIC cxx_std_11)
