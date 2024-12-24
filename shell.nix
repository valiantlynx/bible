{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "python-environment";

  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.dash
    pkgs.python3Packages.plotly
    pkgs.python3Packages.pandas
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scikit-learn
    pkgs.python3Packages.fastapi
    pkgs.python3Packages.debugpy
    pkgs.python3Packages.ollama
    pkgs.python3Packages.uvicorn
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.tqdm
    pkgs.python3Packages.lancedb
    pkgs.jupyter
    pkgs.uv
    pkgs.libuv
    pkgs.pkg-config
    pkgs.stdenv
    pkgs.ollama
  ];

  shellHook = ''
    echo "Welcome to your Python development environment!"
    python --version
  '';
}
