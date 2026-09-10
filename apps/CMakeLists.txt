# CLI front end -- Build Map tickets #4, #5 (seed of #50).
add_executable(sovereign_cli sovereign_cli.cpp)
target_link_libraries(sovereign_cli PRIVATE sovereign::l0)
set_target_properties(sovereign_cli PROPERTIES OUTPUT_NAME sovereign-cli)
if(NOT MSVC)
  target_compile_options(sovereign_cli PRIVATE -Wall -Wextra)
endif()
