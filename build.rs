use std::process::Command;
use std::path::Path;

// RT shaders need a SPIR-V 1.4 target (Vulkan 1.2) for ray tracing extensions.
const RT_SHADERS: &[&str] = &["rt.rgen", "rt.rmiss", "rt_shadow.rmiss", "rt.rchit"];

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("src").join("shaders");

    let shaders = [
        ("shader.vert",    "vert.spv"),
        ("shader.frag",    "frag.spv"),
        ("rt.rgen",        "raygen.spv"),
        ("rt.rmiss",       "miss.spv"),
        ("rt_shadow.rmiss","shadow.spv"),
        ("rt.rchit",       "closesthit.spv"),
    ];

    for (src, dst) in &shaders {
        let src_path = shader_dir.join(src);
        let dst_path = shader_dir.join(dst);

        println!("cargo:rerun-if-changed={}", src_path.display());

        let is_rt = RT_SHADERS.contains(src);

        let mut cmd = Command::new("glslc");
        cmd.arg(&src_path).arg("-o").arg(&dst_path);
        if is_rt {
            cmd.arg("--target-env=vulkan1.2");
        }

        let status = cmd.status();

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
