rtl code of giant accelerator


Giant 使用 45nm FreePDK 综合，ANT、OliVe 数据来源于 paper

BitFusion 数据来源于他们 Repo 中的 8-bit PE，使用 45nm，他们 claim 使用 a commercial standard-cell library 综合

| Architecture | Core            |                  |              | Total Area |
| ------------ | --------------- | ---------------- | ------------ | ---------- |
|              | Component       | Area($\mu m^2$ ) | Number       |            |
| Giant        | 8-bit PE        | 835              | 1024         | 0.860      |
|              | Comparator      | 70               | 32           |            |
|              | Fixed Point Mul | 75               | 32           |            |
| ANT          | 4-bit PE        | 227.34           | 4096         | 0.933      |
|              | Decoder         | 14.00            | 128          |            |
| OliVe        | 4-bit PE        | 227.32           | 4096         | 0.967      |
|              | 4-bit Decoder   | 169.18           | 128          |            |
|              | 8-bit Decoder   | 225              | 64           |            |
| BitFusion    | 8-bit PE        | 687.61           | 1536 (32x48) | 1.056      |

ANT Area 参考，28nm 数据，scale 到 45nm

| 单位 $\mu m^2$ | Component | 28nm  | 45nm   | scaling factor |
| -------------- | --------- | ----- | ------ | -------------- |
| ANT            | 4-bit PE  | 79.57 | 227.34 | 0.35           |
|                | Decoder   | 4.9   | 14     | 0.35           |

OliVe Area 参考，22nm 数据，scale 到 45nm

| 单位 $\mu m^2$ | Component | 28nm  | 45nm   | scaling factor |
| ---- | ---- | ---- | ---- | ---- |
| OliVe | 4-bit PE | 50.01 | 227.32 | 0.22 |
|      | 4-bit Decoder | 37.22 | 169.18 | 0.22 |
|      | 8-bit Decoder | 49.5 | 225 | 0.22 |

Our Component

| Component                            | Area  | Number | Power(uW) |
| ------------------------------------ | ----- | ------ | ----- |
| 8x8 PE (MAC and SAC)                 | 835   | 1024   | 109   | 
| 8x8 PE with bit fusion (MAC and SAC) | 2000  | 1024   | 473   |
| 8x2 PE (MAC and SAC)                 | 495   | 1024   | 90    |
| Comparator, 16 bit float point       | 202   | 32     | 43    |
| Fixed Point Multiplier 4-bit         | 100   | 32     | 45    |
| Fixed Point Multiplier 8-bit         | 380   | 32     | 262   |

