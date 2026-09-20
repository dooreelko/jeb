let
  pkgs = import <nixpkgs> { };

  moth = (builtins.getFlake "github:tailoredshapes/moth")
        .packages.${pkgs.system}.default;

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
    '';
 }
