我核对的是 `vsrc/baselines` 和 `result/baselines`。DC report 不记录源文件路径，所以这是按 `result` 文件名、报告里的 `Design :`、以及 RTL module 名对应出来的。

| 方法 | result 采用的具体 RTL 文件 | report top design |
|---|---|---|
| `ant_olive/ant_pe_*` | `vsrc/baselines/ant_olive/ant_pe.v` | `ant_pe` |
| `ant_olive/ant_pe_fusion_*` | `vsrc/baselines/ant_olive/ant_pe_fusion.v` | `pe_mac_baseline` |
| `ant_olive/ant_pe_fusion_tile_*` | `vsrc/baselines/ant_olive/ant_pe_fusion_tile.v` | `tile_1x4_ant` |
| `ant_olive/pe88_withFusion_withoutShift_*` | `vsrc/baselines/ant_olive/pe88_withFusion_withoutShift.v` | `pe_mac_baseline` |
| `mant/pe82_withFusion_withShift_*` | `vsrc/baselines/mant/pe82_withFusion_withShift.v` | `pe82` |
| `mant/pe88_withFusion_withShift_*` | `vsrc/baselines/mant/pe88_withFusion_withShift.v` | `pe88` |
| `mant/pe88_withFusion_withShift_tile_*` | `vsrc/baselines/mant/pe88_withFusion_withShift_tile.v` | `tile_1x4_mant` |
| `mant/pe88_withoutFusion_withShift_*` | `vsrc/baselines/mant/pe88_withoutFusion_withShift.v` | `pe88_no_fusion` |
| `mant/pe88_withoutFusion_withoutShift_*` | `vsrc/baselines/mant/pe88_withoutFusion_withousShift.v` | `pe88_withFusion_withoutShift` |
| `microscopiq/pe_microscopiq_*` | `vsrc/baselines/microscopiq/pe_microscopiq.v` | `pe_microscopiq` |
| `microscopiq/pe_microscopiq_tile_*` | `vsrc/baselines/microscopiq/pe_microscopiq_tile.v` | `tile_1x4_microscopiq` |
| `tender/tender_pe_*` | `vsrc/baselines/tender/tender_pe.v` | `tender_Pe` |
| `tender/tender_pe_acc16_*` | 当前树里也只能对应 `vsrc/baselines/tender/tender_pe.v` | `tender_Pe` |
| `tender/tender_pe_withFusion_separateRescale_*` | `vsrc/baselines/tender/tender_pe_withFusion_separateRescale.v` | `tender_Pe_withFusion_separateRescale` |
| `tender/tender_pe_withFusion_sharedRescale_*` | `vsrc/baselines/tender/tender_pe_withFusion_sharedRescale.v` | `tender_Pe_withFusion_sharedRescale` |

几个容易混淆的点：

- `*_tile_*` 的 result 采用的是 tile 文件，不是单独的 `_o32.v` 文件；tile 文件内部也定义了 32-bit psum PE。
- `mant/pe88_withoutFusion_withoutShift_*` 对应的 RTL 文件名里有拼写问题：`pe88_withoutFusion_withousShift.v`。
- `tender` 的两个 fusion 文件当前 module 名都写成 `tender_Pe_withFusion`，但历史 report 的 `Design` 是 separate/shared 两个不同名字；说明生成这些 result 时用的 RTL module 名和当前文件内容不完全一致，或当时做过重命名。
