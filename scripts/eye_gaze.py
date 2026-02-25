from typing import Dict, Callable, List
from functools import partial
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.mixins import FeaturesMixin
try:
    from eyelink.objects.geometry import ScreenSetup, \
        CameraSetup, GeometrySetup
except Exception:
    from collections import namedtuple
    GeometrySetup = namedtuple('GeometrySetup', ['vfov', 'hfov'])


class EyeGazeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, gaze_degree_error: float = 1):
        self.gaze_degree_error = gaze_degree_error

        self._features = {}

    def fit(self, X, y=None):
        """
        Fit method required by scikit-learn transformer interface.

        Parameters:
            X (DataFrame): Input data used to check column keys
            y (array-like): Target labels, not used.

        Returns:
            self: Returns the instance itself.
        """
        columns = ['gaze_angle_x', 'gaze_angle_y', 'timestamps']
        for column in columns:
            if column not in X.columns:
                raise ValueError(f"Input dataframe should contain {column=}")
        return self

    def transform(self, X):
        """
        Transform method to extract eye gaze features.

        Parameters:
            X (DataFrame): Input data, extracted

        Returns:
            features (DataFrame): Combined eye gaze features.
        """
        X = X.to_dict('list')
        # todo calculate mean angle and deviation from mean for x and y
        self._add_to_features('angle_x', X['gaze_angle_x'])

        fixations = compute_fixation_durations(X['gaze_angle_x'],
                                               X['gaze_angle_y'],
                                               X['timestamps'],
                                               self.gaze_degree_error)
        self._add_to_features('fixation_duration', fixations)

        saccade_amplitudes = compute_saccade_amplitude(X['gaze_angle_x'],
                                                       X['gaze_angle_y'])
        self._add_to_features('saccade_amplitude', saccade_amplitudes)

        velocities, accelerations = compute_velocity_acceleration(
            X['gaze_angle_x'], X['gaze_angle_y'], X['timestamps']
        )
        self._add_to_features('velocity', velocities)
        self._add_to_features('acceleration', accelerations)

        return self._features

    # def run(self):
    #     self._add_to_features('angle_x', self._gaze_angle_x)
    #
    #     fixations = compute_fixation_durations(self._gaze_angle_x,
    #                                            self._gaze_angle_y,
    #                                            self._timestamps,
    #                                            self._gaze_degree_error)
    #     self._add_to_features('fixation_duration', fixations)
    #
    #     saccade_amplitudes = compute_saccade_amplitude(self._gaze_angle_x,
    #                                                    self._gaze_angle_y)
    #     self._add_to_features('saccade_amplitude', saccade_amplitudes)
    #
    #     velocities, accelerations = compute_velocity_acceleration(
    #         self._gaze_angle_x, self._gaze_angle_x, self._timestamps
    #     )
    #     self._add_to_features('velocity', velocities)
    #     self._add_to_features('acceleration', accelerations)
    #
    #     return self._features

    def _add_to_features(self, name, values):
        mean = np.mean(values)
        std = np.std(values)
        self._features[f'gaze_mean_{name}'] = 0 if np.isnan(mean) else mean
        self._features[f'gaze_std_{name}'] = 0 if np.isnan(std) else std


def compute_fixation_durations(gaze_angle_x, gaze_angle_y,
                               timestamps, threshold):
    """
    Compute fixation durations based on gaze angle data.
    :param gaze_angle_x: List of gaze_angle_x values
    :param gaze_angle_y: List of gaze_angle_y values
    :param timestamps: List of timestamps corresponding to gaze_angle_x and
                        gaze_angle_y
    :param threshold: Angle threshold to consider as a fixation
    :return: List of fixation durations
    """
    fixation_durations = []
    current_fixation_duration = 0

    for i in range(1, len(gaze_angle_x)):
        dx = abs(gaze_angle_x[i] - gaze_angle_x[i - 1])
        dy = abs(gaze_angle_y[i] - gaze_angle_y[i - 1])

        if dx < np.radians(threshold) and dy < np.radians(threshold):
            current_fixation_duration += timestamps[i] - timestamps[i - 1]
        else:
            if current_fixation_duration > 0:
                fixation_durations.append(current_fixation_duration)
                current_fixation_duration = 0

    # Add the last fixation duration if it exists
    if current_fixation_duration > 0:
        fixation_durations.append(current_fixation_duration)

    return fixation_durations


def compute_saccade_amplitude(gaze_angle_x, gaze_angle_y):
    """
    Compute saccade amplitude based on gaze angle data.
    :param gaze_angle_x: List of gaze_angle_x values
    :param gaze_angle_y: List of gaze_angle_y values
    :return: List of saccade amplitudes
    """
    gaze_angles = np.array([gaze_angle_x, gaze_angle_y])
    saccade_amplitudes = np.linalg.norm(np.diff(gaze_angles, axis=1), axis=0)

    # for testing
    # saccade_amplitudes = []
    # for i in range(1, len(gaze_angle_x)):
    #     dx = gaze_angle_x[i] - gaze_angle_x[i-1]
    #     dy = gaze_angle_y[i] - gaze_angle_y[i-1]
    #     amplitude = np.sqrt(dx**2 + dy**2)
    #     saccade_amplitudes.append(amplitude)

    return saccade_amplitudes


def compute_velocity_acceleration(gaze_angle_x, gaze_angle_y, timestamps):
    gaze_angles = np.array([gaze_angle_x, gaze_angle_y])
    # Calculate the velocity and acceleration of gaze movements
    velocity = np.linalg.norm(
        np.diff(gaze_angles, axis=1) / np.diff(timestamps),
        axis=0)
    acceleration = np.diff(velocity, axis=0) / np.diff(timestamps[1:])

    # for testing
    # gaze_velocities = []
    # for i in range(1, len(gaze_angle_x)):
    #     dx = gaze_angle_x[i] - gaze_angle_x[i-1]
    #     dy = gaze_angle_y[i] - gaze_angle_y[i-1]
    #     dt = timestamps[i] - timestamps[i-1]
    #
    #     velocity = np.sqrt(dx**2 + dy**2) / dt
    #     gaze_velocities.append(velocity)
    #
    # gaze_accelerations = []
    # for i in range(1, len(gaze_velocities)):
    #     dv = gaze_velocities[i] - gaze_velocities[i-1]
    #     dt = timestamps[i+1] - timestamps[i]
    #
    #     acceleration = dv / dt
    #     gaze_accelerations.append(acceleration)

    return velocity, acceleration


class AversionExtractor(BaseEstimator, TransformerMixin, FeaturesMixin):
    YAW_COL: str = "gaze_angle_x"
    PITCH_COL: str = "gaze_angle_y"

    def __init__(
            self,
            geometry: GeometrySetup,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        """self.geometry can be any object with .vfov and .hfov properties,
        the easiest way to get it will be:
        from collections import namedtuple

        GeometrySetup = namedtuple('GeometrySetup', ['vfov', 'hfov'])

        geometry = GeometrySetup(vfov=..., hfov=...)
        """
        self.geometry: GeometrySetup = geometry
        self.yaw_col = AversionExtractor.YAW_COL
        self.pitch_col = AversionExtractor.PITCH_COL

        # is needed for redefining properties, e.g. yaw_col or pitch_col
        for name, value in kwargs.items():
            setattr(self, name, value)

    @property
    def features(self) -> Dict[str, Callable]:
        # TODO: here you can change the features
        # you want to calculate and their names
        features: Dict[str, Callable] = {
            # "view_distance_from_center": partial(
            #     AversionExtractor.distance_from_center, center=(0, 0)
            # ),
            "view_looks_at_display": partial(
                AversionExtractor.looks_at_display,
                diameter=self.geometry.vfov, center=(0, 0)
            ),
            # "off_screen_fixations": partial(
            #     AversionExtractor.off_screen_fixations,
            #     looks_at_display_func=partial(
            #         AversionExtractor.looks_at_display,
            #         diameter=self.geometry.vfov, center=(0, 0)
            #     )
            #)
        }
        return features

    @property
    def feature_names(self) -> List[str]:
        return list(self.features.keys())

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, X):
        # 1. Extract yaw pitch in rad in webcamera coordinates
        yaw_pitch = X[[self.yaw_col, self.pitch_col]].values.copy()
        # 2. Transform from webcamera to head coordinate system
        yaw_pitch *= -1
        # 3. Translate center of your by half of vfov
        yaw_pitch[:, 1] = yaw_pitch[:, 1] + self.geometry.vfov / 2
        # 4. Get all features
        self.yaw_pitch = yaw_pitch
        for feature_name, feature_funk in self.features.items():
            self.X[feature_name] = feature_funk(self.yaw_pitch)
        return self.X

    @staticmethod
    def distance_from_center(
        yaw_pitch: np.ndarray,
        center=(0, 0)
    ) -> np.ndarray:
        """
        Calculates the euclidean distance from the center
        returns: np.array([N]), where N is the number of points
        """
        center_x, center_y = center

        # Gaze
        gaze_x, gaze_y = yaw_pitch[:, 0], yaw_pitch[:, 1]

        # Calculate Euclidean distance between gaze point and center
        distance = np.sqrt((gaze_x - center_x) ** 2 + (gaze_y - center_y) ** 2)
        return distance

    @staticmethod
    def looks_at_display(
        yaw_pitch: np.ndarray,
        diameter: float,
        center=(0, 0)
    ) -> np.ndarray:
        distance = AversionExtractor.distance_from_center(yaw_pitch, center)
        distance_greater = distance > diameter / 2
        distance[distance_greater] = 0
        distance[~distance_greater] = 1
        return distance

    @staticmethod
    def off_screen_fixations(
        yaw_pitch: np.ndarray,
        looks_at_display_func: Callable
    ) -> np.ndarray:
        """
        Calculates how many times the gaze goes off the screen
        Note: padding is used to make the array of the same length as the input
        """
        looks = looks_at_display_func(yaw_pitch)
        changes = (looks[:-1] == 1) & (looks[1:] == 0)
        changes_with_padding = np.zeros(len(looks), dtype=bool)
        changes_with_padding[:-1] = changes
        return changes_with_padding.astype(np.int8)

    def plot(self):
        """
        Plots all the points in head-centered CS
        """
        fig, ax = plt.subplots()

        # Rectangle
        anchor_x = - self.geometry.hfov / 2
        anchor_y = - self.geometry.vfov / 2
        rect = patches.Rectangle(
            (anchor_x, anchor_y),
            self.geometry.hfov,
            self.geometry.vfov,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )

        # Circle
        circle = plt.Circle(
            (0, 0),
            self.geometry.vfov / 2,
            color='g',
            fill=False
        )

        ax.set_xlim(- self.geometry.hfov, self.geometry.hfov)
        ax.set_ylim(- self.geometry.vfov, self.geometry.vfov)
        ax.set_xlabel("HFOV")
        ax.set_ylabel("VFOV")

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(circle)
        ax.scatter(
            self.yaw_pitch[:, 0],
            self.yaw_pitch[:, 1],
            c="b",
            label="Sight"
        )
        ax.set_aspect('equal')
        ax.legend()
        ax.grid()
        return fig, ax

    @staticmethod
    def plot_heatmap(
        yaw_pitch_data: list,
        geometry: GeometrySetup,
        bins=100,
        ax=None,
        fig=None,
        title="Heatmap of Gaze Points",
        **kwargs
    ) -> tuple:
        """
        Plots heatmap of the gaze points
        yaw_pitch_data: List of 2D arrays of yaw-pitch points
        geometry: Object containing hfov and vfov values
        bins: Number of bins for the heatmap
        """
        if ax is None or fig is None:
            fig, ax = plt.subplots()
        combined_data = np.vstack(yaw_pitch_data)
        heatmap, xedges, yedges = np.histogram2d(
            combined_data[:, 0], combined_data[:, 1],
            bins=bins,
            density=True,
            range=[[-geometry.hfov, geometry.hfov],
                   [-geometry.vfov, geometry.vfov]
                   ]
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = ax.imshow(
            heatmap.T, extent=extent, origin='lower',
            cmap='viridis',
            norm=LogNorm(
                vmin=kwargs.get("vmin", None),
                vmax=kwargs.get("vmax", None)
            )
        )
        # Add the rectangle and circle
        anchor_x = -geometry.hfov / 2
        anchor_y = -geometry.vfov / 2
        rect = patches.Rectangle(
            (anchor_x, anchor_y),
            geometry.hfov,
            geometry.vfov,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        circle = plt.Circle((0, 0), geometry.vfov / 2, color='g', fill=False)
        ax.add_patch(rect)
        ax.add_patch(circle)

        # Labels and grid
        ax.set_xlim(-geometry.hfov, geometry.hfov)
        ax.set_ylim(-geometry.vfov, geometry.vfov)
        ax.set_xlabel("HFOV")
        ax.set_ylabel("VFOV")
        ax.set_title(title)
        ax.grid()
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Density')

        return fig, ax


if __name__ == "__main__":
    # Example usage
    # All the parameters are taken from Dell Latitude 511, 14“ Bildschirm,
    # Auflösung: 1680x1050, mattes Display.
    try:
        screen = ScreenSetup(
            width=309.9,
            height=174.5,
            resolution_width=1920,
            resolution_height=1080
        )
        camera = CameraSetup(
            vertical_distance_to_monitor=7,  # approximately
            horizontal_distance_to_camera=309.9 / 2
        )
        geometry = GeometrySetup(
            screen_setup=screen,
            camera_setup=camera,
            distance_to_top=600,
            distance_to_bottom=620,  # approximately
        )
    except Exception:
        geometry = GeometrySetup(vfov=0.28880, hfov=0.50545)

    dvpfe = AversionExtractor(geometry)
    df_path = ...
    df = pd.read_csv(df_path)
    df = dvpfe.fit_transform(df)
    df.to_csv(df_path, index=False)
    print('df updated')
