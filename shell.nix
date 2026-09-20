let
  pkgs = import <nixpkgs> { };

  moth = (builtins.getFlake "github:tailoredshapes/moth")
        .packages.${pkgs.system}.default;

in pkgs.mkShellNoCC {
 
    buildInputs = [
      
      pkgs.uv
	  
      moth
   ];
 }


