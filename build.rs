use std::process::Command;
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("src").join("shaders");

    let shaders = [
        ("shader.vert", "vert.spv"),
        ("shader.frag", "frag.spv"),
    ];

    for (src, dst) in &shaders {
        let src_path = shader_dir.join(src);
        let dst_path = shader_dir.join(dst);

        println!("cargo:rerun-if-changed={}", src_path.display());

        let status = Command::new("glslc")
            .arg(&src_path)
            .arg("-o")
            .arg(&dst_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {} -> {}", src, dst);
            }
            Ok(s) => {
                panic!("glslc failed for {} with exit code: {:?}", src, s.code());
            }
            Err(e) => {
                // glslc not found — check if pre-compiled .spv already exists
                if dst_path.exists() {
                    println!(
                        "cargo:warning=glslc not found ({}), using existing {}",
                        e, dst
                    );
                } else {
                    panic!(
                        "glslc not found and no pre-compiled {} exists: {}",
                        dst, e
                    );
                }
            }
        }
    }
}
