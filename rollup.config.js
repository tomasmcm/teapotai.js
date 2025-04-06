import terser from "@rollup/plugin-terser";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

const plugins = (browser) => [
  nodeResolve({ browser }),
  typescript({ 
    compilerOptions: {
      outDir: './dist',
      declarationMap: false,
      declaration: false,
    }
  }),
  terser({ format: { comments: false } })
];

const OUTPUT_CONFIGS = [
  // Node versions
  {
    file: "./dist/teapotai.cjs",
    format: "cjs",
  },
  {
    file: "./dist/teapotai.js",
    format: "esm",
  },

  // Web version
  {
    file: "./dist/teapotai.web.js",
    format: "esm",
  },
];

const WEB_SPECIFIC_CONFIG = {
  onwarn: (warning, warn) => {
    if (!warning.message.includes("@huggingface/transformers")) warn(warning);
  },
};

const NODE_SPECIFIC_CONFIG = {
  external: ["@huggingface/transformers"],
};

export default OUTPUT_CONFIGS.map((output) => {
  const web = output.file.endsWith(".web.js");
  return {
    input: "./src/teapotai.ts",
    output,
    plugins: plugins(web),
    ...(web ? WEB_SPECIFIC_CONFIG : NODE_SPECIFIC_CONFIG),
  };
});
