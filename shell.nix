let
  pkgs = import <nixpkgs> { };

  moth = (builtins.getFlake "github:tailoredshapes/moth")
        .packages.${pkgs.system}.default;

  agg = (builtins.getFlake "github:asciinema/agg").packages.${builtins.currentSystem}.default;

  # nixpkgs has no rocm-all meta package, so list what llama.cpp's HIP backend needs
  # plus the diagnostics (rocminfo, rocm-smi). All of it comes from the binary cache.
  rocm = pkgs.rocmPackages;
  rocmDeps = [
    rocm.clr               # HIP runtime + hipcc driver
    rocm.hipcc
    rocm.rocm-runtime
    rocm.rocm-device-libs
    rocm.rocm-cmake
    rocm.rocblas
    rocm.hipblas
    rocm.hipblaslt
    rocm.rocminfo
    rocm.rocm-smi
  ];

in pkgs.mkShellNoCC {

    buildInputs = [

      pkgs.uv

      moth

      # record the terminal game (asciinema) and turn the recording into a gif (agg)
      pkgs.asciinema
      agg

      # host toolchain for building llama-cpp-python
      pkgs.cmake
      pkgs.gnumake
      pkgs.gcc
    ] ++ rocmDeps;

    # Radeon 860M is gfx1152, which the rocBLAS/hipBLASLt in nixpkgs has no kernels for.
    # Present it as gfx1150 (Strix Point), which they do ship. Build llama.cpp for gfx1150 too.
    HSA_OVERRIDE_GFX_VERSION = "11.5.0";

    ROCM_PATH = "${rocm.clr}";
    HIP_PATH = "${rocm.clr}";

    shellHook = ''
      export CMAKE_PREFIX_PATH="${pkgs.lib.makeSearchPath "" rocmDeps}''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
      # libuuid.so.1: runtime dependency of the NPU-enabled llama.cpp build's XRT backend
      # (scripts/flappy-npu.sh), for xclbin.h-derived symbols. Kept here (not hardcoded in the
      # script) so it doesn't break on nix store GC / rebuilds.
      export LD_LIBRARY_PATH="${pkgs.util-linux.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    '';
 }
