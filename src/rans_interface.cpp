/* Copyright (c) 2021-2024, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *     contributors may be used to endorse or promote products derived from this
 *     software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {


/* Support only 16 bits word max */
inline void Rans64EncPutBits(
    Rans64State *r, uint32_t **pptr, uint32_t val, uint32_t nbits
) {
    assert(nbits <= 16);
    assert(val < (1u << nbits));

    /* Re-normalize */
    uint64_t x = *r;
    uint32_t freq = 1 << (16 - nbits);
    uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t)x;
        x >>= 32;
        Rans64Assert(x < x_max);
    }

    /* x = C(s, x) */
    *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr, uint32_t n_bits) {
    uint64_t x = *r;
    uint32_t val = x & ((1u << n_bits) - 1);

    /* Re-normalize */
    x = x >> n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }

    *r = x;

    return val;
}
} // namespace


void BufferedRansEncoder::encode_with_numpy(
    const py::array_t<int32_t> &symbols,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets
) {
    assert(symbols.shape(0) == indexes.shape(0));
    assert(cdfs.shape(0) == cdfs_sizes.shape(0) && cdfs_sizes.shape(0) == offsets.shape(0));

    auto u_symbols = symbols.unchecked<1>();
    auto u_indexes = indexes.unchecked<1>();
    auto u_cdfs = cdfs.unchecked<2>();
    auto u_cdfs_sizes = cdfs_sizes.unchecked<1>();
    auto u_offsets = offsets.unchecked<1>();

    size_t num_symbols = u_symbols.shape(0);
    size_t num_cdfs = u_cdfs.shape(0);
    size_t vocab_size = u_cdfs.shape(1);

    // backward loop on symbols from the end;
    for (size_t i = 0; i < num_symbols; ++i) {
        const int32_t cdf_idx = u_indexes(i);
        assert(0 <= cdf_idx && cdf_idx < num_cdfs);

        const int32_t max_value = u_cdfs_sizes(cdf_idx) - 2;
        assert(0 <= max_value && max_value < vocab_size - 1);

        int32_t value = u_symbols(i) - u_offsets(cdf_idx);

        uint32_t raw_val = 0;
        if (value < 0) {
            // std::cout << "Warning: 0 > value == " << value << std::endl;
            raw_val = -2 * value - 1;
            value = max_value;
        } else if (value >= max_value) {
            // std::cout << "Warning: value >= max_value: " << value << " > " << max_value << std::endl;
            raw_val = 2 * (value - max_value);
            value = max_value;
        }

        assert(0 <= value && value < u_cdfs_sizes(cdf_idx) - 1);

        _syms.push_back({
            static_cast<uint16_t>(u_cdfs(cdf_idx, value)),
            static_cast<uint16_t>(u_cdfs(cdf_idx, value + 1) - u_cdfs(cdf_idx, value)),
            false
        });

        /* Bypass coding mode (value == max_value -> sentinel flag) */
        if (value == max_value) {
            // std::cout << "Warning: value == max_value == " << value << std::endl;
            /* Determine the number of bypasses (in bypass_precision size) needed to
             * encode the raw value. */
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
                ++n_bypass;
            }

            /* Encode number of bypasses */
            int32_t val = n_bypass;
            while (val >= max_bypass_val) {
                _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
                val -= max_bypass_val;
            }
            _syms.push_back(
                    {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

            /* Encode raw value */
            for (int32_t j = 0; j < n_bypass; ++j) {
                const int32_t val =
                        (raw_val >> (j * bypass_precision)) & max_bypass_val;
                _syms.push_back(
                        {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
            }
        }
    }
}


py::bytes BufferedRansEncoder::flush() {
    Rans64State rans;
    Rans64EncInit(&rans);

    std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
    uint32_t *ptr = output.data() + output.size();
    assert(ptr != nullptr);

    while (!_syms.empty()) {
        const RansSymbol sym = _syms.back();

        if (!sym.bypass) {
            Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
        } else {
            // unlikely...
            Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        }
        _syms.pop_back();
    }

    Rans64EncFlush(&rans, &ptr);

    const int nbytes = std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
    return std::string(reinterpret_cast<char *>(ptr), nbytes);
}


py::bytes RansEncoder::encode_with_numpy(
    const py::array_t<int32_t> &symbols,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets
) {
    BufferedRansEncoder buffered_rans_enc;
    buffered_rans_enc.encode_with_numpy(symbols, indexes, cdfs, cdfs_sizes, offsets);
    return buffered_rans_enc.flush();
}


uint32_t rfind(
    const py::array_t<int32_t> &cdfs,
    const int32_t row_idx,
    const int32_t max_length,
    const uint32_t cum_freq
) {
    assert(0 <= row_idx && row_idx < cdfs.shape(0));
    assert(0 <= max_length && max_length < cdfs.shape(1));

    auto u_cdfs = cdfs.unchecked<2>();

    for (size_t i = 0; i < max_length; ++i) {
        if (u_cdfs(row_idx, i) > cum_freq) {return i - 1;}
    }
    return max_length - 1;
}


py::array_t<int32_t> RansDecoder::decode_with_numpy(
    const std::string &encoded,
    const py::array_t<int32_t> &indexes,
    const py::array_t<int32_t> &cdfs,
    const py::array_t<int32_t> &cdfs_sizes,
    const py::array_t<int32_t> &offsets
) {
    assert(cdfs.shape(0) == cdfs_sizes.shape(0) && cdfs_sizes.shape(0) == offsets.shape(0));

    auto u_indexes = indexes.unchecked<1>();
    auto u_cdfs = cdfs.unchecked<2>();
    auto u_cdfs_sizes = cdfs_sizes.unchecked<1>();
    auto u_offsets = offsets.unchecked<1>();

    size_t num_symbols = u_indexes.shape(0);
    size_t num_cdfs = u_cdfs.shape(0);
    size_t vocab_size = u_cdfs.shape(1);

    py::array_t<int32_t> output(num_symbols);

    Rans64State rans;
    uint32_t *ptr = (uint32_t *)encoded.data();
    assert(ptr != nullptr);
    Rans64DecInit(&rans, &ptr);

    for (size_t i = 0; i < static_cast<int>(num_symbols); ++i) {
        const int32_t cdf_idx = u_indexes(i);
        assert(0 <= cdf_idx && cdf_idx < num_cdfs);

        int32_t size = u_cdfs_sizes(cdf_idx);

        const int32_t max_value = size - 2;
        assert(0 <= max_value && max_value < vocab_size - 1);

        const int32_t offset = u_offsets(cdf_idx);

        const uint32_t cum_freq = Rans64DecGet(&rans, precision);

        const uint32_t s = rfind(cdfs, cdf_idx, size, cum_freq);

        // Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);
        auto cdf_left = u_cdfs(cdf_idx, s);
        auto cdf_right = u_cdfs(cdf_idx, s + 1);
        Rans64DecAdvance(&rans, &ptr, cdf_left, cdf_right - cdf_left, precision);

        int32_t value = static_cast<int32_t>(s);

        if (value == max_value) {
            // std::cout << "Warning: value == max_value == " << value << std::endl;
            /* Bypass decoding mode */
            int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
            int32_t n_bypass = val;

            while (val == max_bypass_val) {
                val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                n_bypass += val;
            }

            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
                assert(val <= max_bypass_val);
                raw_val |= val << (j * bypass_precision);
            }
            value = raw_val >> 1;
            if (raw_val & 1) {
                value = -value - 1;
            } else {
                value += max_value;
            }
        }

        output.mutable_at(i) = value + offset;
    }

    return output;
}


PYBIND11_MODULE(ansnp, m) {
    m.doc() = "range Asymmetric Numeral System python bindings";

    py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder", py::module_local())
            .def(py::init<>())
            .def("flush", &BufferedRansEncoder::flush);

    py::class_<RansEncoder>(m, "RansEncoder", py::module_local())
            .def(py::init<>())
            .def("encode_with_numpy", &RansEncoder::encode_with_numpy);

    py::class_<RansDecoder>(m, "RansDecoder", py::module_local())
            .def(py::init<>())
            .def("decode_with_numpy", &RansDecoder::decode_with_numpy);
}
