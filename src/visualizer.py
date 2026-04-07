# <PROJECT NAME>
# Copyright (C) <2025>  <Yutaro Hara>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.

import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt


class SimulationGUI:
    """
    Dear PyGui-based GUI for interactive wave simulation visualization.

    Provides:
    - Play / Pause toggle
    - Dynamic isnap (update interval) slider
    - Colormap selector (seismic, jet, viridis, ...)
    - Real-time step counter and calculation speed display
    - U / V / W wavefield texture panels

    Usage::

        gui = SimulationGUI(nx, nz, dx, dz)
        gui.isnap = sim.isnap          # sync initial value
        gui.setup(nt)                  # create context & viewport

        while gui.is_running():
            if gui.playing and it < nt:
                # advance physics one step …
                if it % gui.isnap == 0:
                    gui.update_textures(u_cpu, v_cpu, w_cpu)
                    gui.update_status(it, ms_per_step)
                it += 1
            gui.render_frame()

        gui.cleanup()
    """

    COLORMAPS = ['seismic', 'jet', 'viridis', 'RdBu', 'bwr', 'hot']

    def __init__(self, nx: int, nz: int, dx: float, dz: float):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz

        self.playing: bool = True
        self.isnap: int = 10
        self.colormap: str = 'seismic'

        self._nt: int = 0
        self._cmap_cache: dict = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, nt: int) -> None:
        """Create DPG context, textures, windows, and show the viewport."""
        self._nt = nt
        dpg.create_context()

        # Blank RGBA texture — height=nz rows, width=nx cols (transposed layout)
        blank = np.zeros(self.nz * self.nx * 4, dtype=np.float32)

        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=self.nx, height=self.nz,
                default_value=blank,
                format=dpg.mvFormat_Float_rgba,
                tag='tex_u',
            )
            dpg.add_raw_texture(
                width=self.nx, height=self.nz,
                default_value=blank,
                format=dpg.mvFormat_Float_rgba,
                tag='tex_v',
            )
            dpg.add_raw_texture(
                width=self.nx, height=self.nz,
                default_value=blank,
                format=dpg.mvFormat_Float_rgba,
                tag='tex_w',
            )

        # Scale display images: fixed height=300, width proportional to grid aspect
        img_h = 300
        img_w = max(1, int(img_h * self.nx / max(1, self.nz)))

        with dpg.window(
            label='Wave Simulation',
            tag='main_win',
            width=img_w * 3 + 370,
            height=img_h + 120,
            no_close=True,
        ):
            with dpg.group(horizontal=True):
                # ---- Wavefield image panels ----
                with dpg.group():
                    dpg.add_text('U Wavefield')
                    dpg.add_image('tex_u', width=img_w, height=img_h)
                with dpg.group():
                    dpg.add_text('V Wavefield')
                    dpg.add_image('tex_v', width=img_w, height=img_h)
                with dpg.group():
                    dpg.add_text('W Wavefield')
                    dpg.add_image('tex_w', width=img_w, height=img_h)

                dpg.add_spacer(width=12)

                # ---- Control / status panel ----
                with dpg.group():
                    dpg.add_button(
                        label='Pause',
                        tag='btn_playpause',
                        callback=self._toggle_play,
                        width=150,
                    )
                    dpg.add_separator()

                    dpg.add_text('Update Interval (isnap):')
                    dpg.add_slider_int(
                        tag='slider_isnap',
                        default_value=self.isnap,
                        min_value=1,
                        max_value=200,
                        width=190,
                        callback=self._on_isnap_change,
                    )
                    dpg.add_separator()

                    dpg.add_text('Colormap:')
                    dpg.add_combo(
                        items=self.COLORMAPS,
                        tag='combo_cmap',
                        default_value=self.colormap,
                        width=190,
                        callback=self._on_cmap_change,
                    )
                    dpg.add_separator()

                    dpg.add_text('--- Status ---')
                    dpg.add_text(f'Step: 0 / {nt}', tag='lbl_step')
                    dpg.add_text('Speed: -- ms/step', tag='lbl_speed')

        dpg.create_viewport(
            title='Wave Simulation (DPG)',
            width=img_w * 3 + 390,
            height=img_h + 160,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        dpg.set_item_label('btn_playpause', 'Pause' if self.playing else 'Play')

    def _on_isnap_change(self, sender, app_data) -> None:  # noqa: ARG002
        self.isnap = int(app_data)

    def _on_cmap_change(self, sender, app_data) -> None:  # noqa: ARG002
        self.colormap = app_data

    # ------------------------------------------------------------------
    # Texture helpers
    # ------------------------------------------------------------------

    def _get_cmap(self, name: str):
        if name not in self._cmap_cache:
            self._cmap_cache[name] = plt.get_cmap(name)
        return self._cmap_cache[name]

    def _to_rgba(self, arr_cpu: np.ndarray) -> np.ndarray:
        """
        Convert an (nx, nz) float32 ndarray to a flat float32 RGBA array
        suitable for a DPG raw texture (height=nz, width=nx layout).
        """
        # Transpose so rows = z-axis (depth), cols = x-axis
        arr_T = np.ascontiguousarray(arr_cpu.T, dtype=np.float32)  # (nz, nx)
        amax = float(np.abs(arr_T).max())
        if amax > 0.0:
            arr_T /= amax
        # Map symmetric range [-1, 1] -> [0, 1] for colormap lookup
        arr_01 = np.clip((arr_T + 1.0) * 0.5, 0.0, 1.0)
        rgba = self._get_cmap(self.colormap)(arr_01)   # (nz, nx, 4) float64
        return np.ascontiguousarray(rgba, dtype=np.float32).flatten()

    def update_textures(
        self,
        u_cpu: np.ndarray,
        v_cpu: np.ndarray,
        w_cpu: np.ndarray,
    ) -> None:
        """Push new wavefield numpy arrays to the three DPG textures."""
        dpg.set_value('tex_u', self._to_rgba(u_cpu))
        dpg.set_value('tex_v', self._to_rgba(v_cpu))
        dpg.set_value('tex_w', self._to_rgba(w_cpu))

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def update_status(self, step: int, speed_ms: float) -> None:
        dpg.set_value('lbl_step', f'Step: {step} / {self._nt}')
        dpg.set_value('lbl_speed', f'Speed: {speed_ms:.2f} ms/step')

    # ------------------------------------------------------------------
    # Main loop helpers
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return True while the DPG viewport is open."""
        return dpg.is_dearpygui_running()

    def render_frame(self) -> None:
        """Render a single DPG frame (process events + draw)."""
        dpg.render_dearpygui_frame()

    def cleanup(self) -> None:
        """Destroy the DPG context and release all resources."""
        dpg.destroy_context()
