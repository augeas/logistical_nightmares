import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        ## Logistical Nightmares

        ### (In which we abuse [bifurcation diagrams](https://en.wikipedia.org/wiki/Bifurcation_diagram) and the [inverse discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms).)

        Consider the [Logistic Map](https://en.wikipedia.org/wiki/Logistic_map):

        $$\Large{x_{n+1}=rx_{n}}(1-x_{n})$$

        $x$ is bound between 0 and 1 if $r$ is in the range 0 to 4. Scan values of $r$ on the horizontal axis, and plot the trajectory of $x$ on the vertical axis after discarding a few intitial transients, and the familiar bifurcation diagram results. (It was popularlized by [Robert May's](https://en.wikipedia.org/wiki/Robert_May,_Baron_May_of_Oxford) [1976 paper in Nature](https://www.researchgate.net/publication/237005499_Simple_Mathematical_Models_With_Very_Complicated_Dynamics).)
        """
    )
    return


@app.cell
def _():
    from datetime import datetime
    from functools import partial
    from io import BufferedReader, BytesIO
    import os
    import subprocess as sp
    import sys

    import marimo as mo
    try:
        from numba import njit
    except:
        njit = None
    import numpy as np
    from PIL import Image
    from scipy.fftpack import idct
    from scipy.io import wavfile
    from scipy.signal.windows import blackmanharris
    return (
        BufferedReader,
        BytesIO,
        Image,
        blackmanharris,
        datetime,
        idct,
        mo,
        njit,
        np,
        os,
        partial,
        sp,
        sys,
        wavfile,
    )


@app.cell
def _(np):
    def symbol_seq(seq, size):
        symbols = np.fromiter(map(ord, seq.upper()), dtype=int) - 65
        if size % len(seq) == 0:
            reps = size // len(seq)
        else:
            reps = size // len(seq) + 1
        return np.tile(symbols, reps)[0:size]
    return (symbol_seq,)


@app.cell
def _(np):
    def rot_coeffs(x, y, radius, n):
        theta = np.linspace(-np.pi, np.pi, n)
        rot = np.zeros(shape=(n, 2, 2))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot[:, 0, 0] = cos_theta
        rot[:, 0, 1] = sin_theta
        rot[:, 1, 0] = -sin_theta
        rot[:, 1, 1] = cos_theta
        point = np.zeros(shape=(1, 1, 2))
        point[:, :, -1] = radius
        return (np.array([x, y]).reshape((1, 1, 2)) + point @ rot).reshape(n, 2)
    return (rot_coeffs,)


@app.cell
def _(np, rot_coeffs):
    def double_rot_coeffs(x1, y1, r1, n1, x2, y2, r2, n2):
        coeffs = np.zeros((n1 * n2, 4))
        coeffs[:, 0:2] = np.tile(rot_coeffs(x1, y1, r1, n1), [n2, 1])
        for i, row in enumerate(rot_coeffs(x2, y2, r2, n2)):
            idx = i * n1
            coeffs[idx:idx+n1, 2:] = row
        return coeffs
    return (double_rot_coeffs,)


@app.cell
def _(np, rot_coeffs):
    def double_rot_coeffs_iter(x1, y1, r1, n1, x2, y2, r2, n2):
        coeffs = np.zeros((n1, 4))
        coeffs[:, 0:2] = rot_coeffs(x1, y1, r1, n1)
        for row in rot_coeffs(x2, y2, r2, n2):
            coeffs[:, 2:] = row
            yield coeffs
    return (double_rot_coeffs_iter,)


@app.cell
def _(njit, np):
    def bifucrcation_histogram(its, bins):
        hist, _ = np.histogram(its, bins, (0, 1))
        return hist / its.shape[0]

    if njit is None:
        bifucrcation_hist = bifucrcation_histogram
    else:
        bifucrcation_hist = njit(bifucrcation_histogram)
    return bifucrcation_hist, bifucrcation_histogram


@app.cell
def _(njit, np):
    def logistic_f(coeffs, seq, bins, n, skip):
        its = np.zeros(shape=(coeffs.shape[0], n))
        its[:, 0] = 0.5
        for i, j, idx in zip(range(1, skip), range(skip-1), seq[0:skip]):
            its[:, i] = coeffs[:, idx] * its[:, j] * (1.0 - its[:, j])
        its[:, 0] = its[:, skip-1]
        for i, j, idx, in zip(range(1, n), range(n-1), seq[skip:]):
            its[:, i] = coeffs[:, idx] * its[:, j] * (1.0 - its[:, j])
        return its

    if njit is None:
        logistic = logistic_f
    else:
        logistic = njit(logistic_f)
    return logistic, logistic_f


@app.cell
def _(bifucrcation_hist, logistic, np):
    def logistic_map(coeffs, seq, bins, n, skip):
        return np.apply_along_axis(
            bifucrcation_hist, 1, logistic(coeffs, seq, bins, n, skip), bins
        )
    return (logistic_map,)


@app.cell
def _(Image, np):
    def render_img(hist):
        img = np.zeros(shape=hist.T.shape+(3,), dtype=np.uint8)
        img[::-1, :, 1] = (255 * np.cbrt(hist.T)).astype(np.uint8)
        return Image.fromarray(img)
    return (render_img,)


@app.cell
def _(logistic_map, np, render_img):
    def basic_map(start=2.95, end=4.0, width=360, height=240):
        points = 8 * height
        return render_img(logistic_map(
            np.linspace(start, end, width).reshape(width, 1),
            np.zeros(128+points, dtype=int),
            height, points, 128
        ))
    return (basic_map,)


@app.cell
def _(basic_map, mo):
    mo.center(basic_map())
    return


@app.cell
def _(mo):
    mo.md(
        """
        One can extend the Logistic Map by replacing $r$ with some repeating sequence of coefficients $AB$. (This has been used to construct
        [Lyapunov Fractals](https://en.wikipedia.org/wiki/Lyapunov_fractal).) Consider a circle on the $AB$ plane. Each point on the circle corresponds to a pair of values for the coefficients. The horizontal axis of the bifurcation diagram now corresponds to rotation around the circle, so it ends where it begins in a cycle. We might as well use sequences of four coefficients, $ABCD$. This time, for every point on a circle in the $CD$ plane, there is one rotation in the $AB$ plane. Rotation of the coefficients in the $CD$ plane causes the entire diagram to shift and change at once. Between the chosen sequence and the radii and centres of the two circles, the form of the bifurcation diagram can vary considerably:
        """
    )
    return


@app.cell
def _(mo):
    UI_IMG_WIDTH = 640
    UI_IMG_HEIGHT = 480
    UI_POINTS = 8 * UI_IMG_HEIGHT
    UI_TOTAL_POINTS = 128 + UI_POINTS
    seq_box = mo.ui.text(value='AACDCBB', label='sequence')
    ab_r_slider = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.25, label='AB radius')
    cd_r_slider = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.25, label='CD radius')
    return (
        UI_IMG_HEIGHT,
        UI_IMG_WIDTH,
        UI_POINTS,
        UI_TOTAL_POINTS,
        ab_r_slider,
        cd_r_slider,
        seq_box,
    )


@app.cell
def _(ab_r_slider, mo):
    ab_max = 4.0 - ab_r_slider.value
    ab_val = (2.8 + ab_max) / 2
    a_cen_slider = mo.ui.slider(start=2.8, stop=ab_max, step=0.05, value=ab_val, label='A centre')
    b_cen_slider = mo.ui.slider(start=2.8, stop=ab_max, step=0.05, value=ab_val, label='B centre')
    return a_cen_slider, ab_max, ab_val, b_cen_slider


@app.cell
def _(cd_r_slider, mo):
    cd_max = 4.0 - cd_r_slider.value
    cd_val = (2.8 + cd_max) / 2
    c_cen_slider = mo.ui.slider(start=2.8, stop=cd_max, step=0.05, value=cd_val, label='C centre')
    d_cen_slider = mo.ui.slider(start=2.8, stop=cd_max, step=0.05, value=cd_val, label='D centre')
    return c_cen_slider, cd_max, cd_val, d_cen_slider


@app.cell
def _(UI_TOTAL_POINTS, seq_box, symbol_seq):
    ui_seq = symbol_seq(
        ''.join(filter(lambda c: c in 'ABCD', seq_box.value.upper())),
        UI_TOTAL_POINTS
    )
    return (ui_seq,)


@app.cell
def _(mo):
    phase_slider = mo.ui.slider(start=0, stop=49, step=1, value=25, label='CD angle')
    return (phase_slider,)


@app.cell
def _(c_cen_slider, cd_r_slider, d_cen_slider, rot_coeffs):
    cd_coeffs = rot_coeffs(c_cen_slider.value, d_cen_slider.value, cd_r_slider.value, 50)
    return (cd_coeffs,)


@app.cell
def _(
    UI_IMG_WIDTH,
    a_cen_slider,
    ab_r_slider,
    b_cen_slider,
    cd_coeffs,
    np,
    phase_slider,
    rot_coeffs,
):
    ui_coeffs = np.zeros((UI_IMG_WIDTH, 4))
    ui_coeffs[:, 0:2] = rot_coeffs(a_cen_slider.value, b_cen_slider.value, ab_r_slider.value, UI_IMG_WIDTH)
    ui_coeffs[:, 2:] = cd_coeffs[phase_slider.value]
    return (ui_coeffs,)


@app.cell
def _(
    UI_IMG_HEIGHT,
    UI_TOTAL_POINTS,
    logistic_map,
    render_img,
    ui_coeffs,
    ui_seq,
):
    ui_img = render_img(logistic_map(ui_coeffs, ui_seq, UI_IMG_HEIGHT, UI_TOTAL_POINTS, 128))
    return (ui_img,)


@app.cell
def _(
    a_cen_slider,
    ab_r_slider,
    b_cen_slider,
    c_cen_slider,
    cd_r_slider,
    d_cen_slider,
    mo,
    phase_slider,
    seq_box,
    ui_img,
):
    ui = mo.vstack([
        ui_img,
        mo.hstack([
            mo.vstack([ab_r_slider, a_cen_slider, b_cen_slider]),
            mo.vstack([cd_r_slider, c_cen_slider, d_cen_slider]),
            mo.vstack([seq_box, phase_slider])
        ], justify='center')
    ], align='center')
    return (ui,)


@app.cell
def _(ui):
    ui
    return


@app.cell
def _(njit, np):
    def pan_histogram(hist, rnge):
        return np.average(rnge, weights=hist) / len(hist)

    if njit is None:
        pan_hist = pan_histogram
    else:
        pan_hist = njit(pan_histogram)
    return pan_hist, pan_histogram


@app.cell
def _(blackmanharris, idct, np, pan_hist):
    def bifurcation_noise(hist, seg_size, hop=4, pan=0.5, compress=False, mono=False):
        n_segs = hist.shape[0]
        hop_size = seg_size // hop
        n_hops = n_segs - 1 + hop - 1
        total_samples = n_hops * hop_size
        segments = idct(
            np.cbrt(hist) if compress else hist, n=seg_size, axis=1
        ) * blackmanharris(seg_size)
        if not mono:
            pan_sig =  np.apply_along_axis(pan_hist, 1, hist, np.arange(hist.shape[-1])).reshape((n_segs, 1))
            bias = 1 - pan
            stereo_segments = np.zeros((n_segs, seg_size, 2))
            stereo_segments[:, :, 0] = (bias + pan * pan_sig) * segments
            stereo_segments[:, :, 1] = (bias + pan * (1 - pan_sig)) * segments
            all_segments = stereo_segments
            samples = np.zeros((total_samples, 2))
        else:
            all_segments = segments
            samples = np.zeros(total_samples)

        samples = np.zeros((total_samples, 2))
        for i, seg in enumerate(all_segments[0:n_segs-hop+1]):
            start = i * hop_size
            end = start + seg_size
            samples[start:end, :] += seg
        offset = total_samples  - hop_size * (hop - 1) 
        for i, seg in enumerate(all_segments[n_segs-hop+1:]):
            seg_end = (hop - i - 1) * hop_size
            sample_start = offset + i * hop_size
            samples[sample_start:] += seg[0:seg_end]
            sample_end = seg_size - seg_end
            samples[0:sample_end] += seg[seg_end:]
        return samples
    return (bifurcation_noise,)


@app.cell
def _(np):
    def norm_samples(samples, pcm=False, short=False):
        sample_min = samples.min(axis=0)
        sample_range = samples.max(axis=0) - sample_min
        norm = (2 * (samples - sample_min) / sample_range).astype(np.float32) - np.ones((1, 2), dtype=np.float32)
        if not pcm:
            return norm
        else:
            if not short:
                return (32767 * norm).astype(np.int16)
            else:
                return (127 + norm * 127).astype(np.uint8)
    return (norm_samples,)


@app.cell
def _(mo):
    mo.md(
        """
        The intensity of each point on the bifurcation diagram is found by taking a histogram of the points in each column generated by the logistic map for the given coefficients. More frequently visited points are brighter. Consider each histogram as a series of spectra that can be turned into audio by taking the inverse [discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform). The audio segments are multiplied by a [window function](https://en.wikipedia.org/wiki/Window_function), then overlapped and summed. Each segment is shifted by a hop size. For each point on the $CD$ circle, a bifurcation diagram and sequence of audio segments is produced for one complete rotation around the $AB$ circle. The last segment overlaps the first, so the audio can loop endlessly. The "centre of mass" of each histogram, the point with equal area either side of it, can be used to pan the signal.

        The number of segments will be determined by the product of the numbers of points around the $AB$ and $CD$ circles. The number of bins in each histogram can be set relative to the selected discrete cosine transform size. The smaller the number of bins, the more zero-padding, and the lower the pitch of the sound. The hop-size is also set as a fraction of the number of samples in each segment. When you have finished adjusting the parameters, press the button to generate a `.wav` file.

        (If this is being run via [WASM](https://docs.marimo.io/guides/wasm/), the maximum allowed output size is rather small, only very short loops can be generated unless 8-bit samples at a low rate are used.)
        """
    )
    return


@app.cell
def _(mo, sys):
    if sys.platform == 'emscripten':
        DEF_SEG_SIZE = '1024'
        DEF_AUDIO_RATE = '22kHz'
        DEF_MONO = True
        BIT_DEPTH = '8'
    else:
        DEF_SEG_SIZE = '8192'
        DEF_AUDIO_RATE = '44.1kHz'
        DEF_MONO = False
        BIT_DEPTH = '32'

    ab_points_slider = mo.ui.slider(start=24, stop=120, step=2, value=60, label='AB points')
    cd_points_slider = mo.ui.slider(start=24, stop=120, step=2, value=48, label='CD points')
    bins_dropdown = mo.ui.dropdown(
        {'1/{}'.format(i): i for i in map(lambda x: 2**x, range(2,8))},
        value='1/32', label='bins'
    )
    seg_size_dropdown = mo.ui.dropdown(
        {str(2**i): i for i in range(10,17)}, value=DEF_SEG_SIZE, label='segment size'
    )
    hop_size_dropdown = mo.ui.dropdown(
        {'1/{}'.format(i): i for i in map(lambda x: 2**x, range(1,4))},
        value='1/4', label='hop size'
    )
    pan_slider = mo.ui.slider(start=0, stop=1.0, step=0.05, value=0.75, label='pan depth')
    sample_rate_dropdown = mo.ui.dropdown(
        {'{}kHz'.format(r): int(r*1000) for r in (16, 22, 44.1, 48)},
        value=DEF_AUDIO_RATE, label='sample rate'
    )
    bit_depth_dropdown = mo.ui.dropdown(
        {str(d): d for d in (32, 16, 8)},
        value=BIT_DEPTH, label='bit depth'
    )
    mono_box = mo.ui.checkbox(value=DEF_MONO, label='mono')
    compress_box = mo.ui.checkbox(value=False, label='compress')
    return (
        BIT_DEPTH,
        DEF_AUDIO_RATE,
        DEF_MONO,
        DEF_SEG_SIZE,
        ab_points_slider,
        bins_dropdown,
        bit_depth_dropdown,
        cd_points_slider,
        compress_box,
        hop_size_dropdown,
        mono_box,
        pan_slider,
        sample_rate_dropdown,
        seg_size_dropdown,
    )


@app.cell
def _(
    ab_points_slider,
    cd_points_slider,
    hop_size_dropdown,
    np,
    sample_rate_dropdown,
    seg_size_dropdown,
):
    def track_duration():
        n_segs = ab_points_slider.value * cd_points_slider.value
        seg_size = 2**seg_size_dropdown.value
        hop_size = seg_size // hop_size_dropdown.value
        n_hops = n_segs - 1 + hop_size_dropdown.value - 1
        total_samples = n_hops * hop_size
        return total_samples

    def format_duration(samples):
        duration = np.rint(samples / sample_rate_dropdown.value).astype(int)
        mins = duration // 60
        secs = duration % 60
        return 'duration: {}:{}'.format(mins, ('0'+str(secs))[-2:])
    return format_duration, track_duration


@app.cell
def _(
    BytesIO,
    a_cen_slider,
    ab_points_slider,
    ab_r_slider,
    b_cen_slider,
    bifurcation_noise,
    bins_dropdown,
    bit_depth_dropdown,
    c_cen_slider,
    cd_points_slider,
    cd_r_slider,
    compress_box,
    d_cen_slider,
    double_rot_coeffs,
    hop_size_dropdown,
    logistic_map,
    norm_samples,
    pan_slider,
    sample_rate_dropdown,
    seg_size_dropdown,
    seq_box,
    symbol_seq,
    wavfile,
):
    def build_ui_track(_):
        tk_coeffs = double_rot_coeffs(
            a_cen_slider.value, b_cen_slider.value, ab_r_slider.value, ab_points_slider.value,
            c_cen_slider.value, d_cen_slider.value, cd_r_slider.value, cd_points_slider.value,
        )

        seg_size = 2**seg_size_dropdown.value
        bins = seg_size // bins_dropdown.value
        tk_points = 8 * bins

        tk_seq = symbol_seq(
            ''.join(filter(lambda c: c in 'ABCD', seq_box.value.upper())),
            tk_points + 128
        )

        tk_hist = logistic_map(tk_coeffs, tk_seq, bins, tk_points, 128)

        if bit_depth_dropdown.value == 32:
            pcm = False
            short = False
        else:
            pcm = True
            short = bit_depth_dropdown.value == 8

        with BytesIO() as bfile:
            wavfile.write(
                bfile, sample_rate_dropdown.value,
                norm_samples(
                    bifurcation_noise(
                        tk_hist, seg_size,
                        hop_size_dropdown.value, pan_slider.value,
                        compress_box.value
                    ), pcm=pcm, short=short
                )
            )
            return bfile.getvalue()
    return (build_ui_track,)


@app.cell
def _(build_ui_track, mo):
    audio_button = mo.ui.button(value=bytes(1), on_click=build_ui_track, label='generate audio')
    return (audio_button,)


@app.cell
def _(audio_button, mo):
    audio_player = mo.audio(audio_button.value)
    return (audio_player,)


@app.cell
def _(audio_button, mo, seq_box):
    wav_download = mo.download(
        data=audio_button.value,
        filename='{}.wav'.format(seq_box.value),
        mimetype='audio/wav',
    )
    return (wav_download,)


@app.cell
def _(
    ab_points_slider,
    audio_button,
    audio_player,
    bins_dropdown,
    bit_depth_dropdown,
    cd_points_slider,
    compress_box,
    format_duration,
    hop_size_dropdown,
    mo,
    mono_box,
    pan_slider,
    sample_rate_dropdown,
    seg_size_dropdown,
    track_duration,
    wav_download,
):
    audio_ui = mo.vstack([
        mo.hstack([
            mo.vstack([ab_points_slider, cd_points_slider, bins_dropdown]),
            mo.vstack([seg_size_dropdown, hop_size_dropdown, pan_slider])
        ], justify='center'),
        mo.hstack([
            sample_rate_dropdown, bit_depth_dropdown, mono_box, compress_box
        ]),
        mo.hstack([
            audio_button, format_duration(track_duration())
        ]),    
        audio_player, wav_download
    ], align='center')
    return (audio_ui,)


@app.cell
def _(audio_ui):
    audio_ui
    return


@app.cell
def _(
    double_rot_coeffs_iter,
    logistic_map,
    partial,
    render_img,
    symbol_seq,
):
    def vid_seq(seq, x1, y1, r1, x2, y2, r2, n_frames, width=1080, height=1080):
        coeff = double_rot_coeffs_iter(x1, y1, r1, width, x2, y2, r2, n_frames)
        n_points = 8 * height
        seq = symbol_seq(seq, size=n_points+128)
        lm = partial(logistic_map, seq=seq, bins=height, n=n_points, skip=128)
        yield from map(render_img, map(lm, coeff))
    return (vid_seq,)


@app.cell
def _(mo, vid_seq):
    def vid_seq_bar(*args, **kwargs):
        with mo.status.progress_bar(total=args[7], remove_on_exit=True) as bar:
            for im in vid_seq(*args, **kwargs):
                yield im
                bar.update()
    return (vid_seq_bar,)


@app.cell
def _(sp):
    def render_video(audio_name, video_name, im_seq, fps=30):
        ffmpeg_cmd = [
            'ffmpeg', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps),
            '-i', '-', '-i', audio_name,
            '-vcodec', 'libx264',  '-c:a', 'aac', '-q:v', '0', video_name
        ]
        pipe = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        for im in im_seq:
            im.save(pipe.stdin, 'PNG')
        pipe.stdin.close()
        pipe.wait()
    return (render_video,)


@app.cell
def _(mo):
    VID_SIZES = ('426x240', '640x480', '512x512', '1280x720', '1024x1024', '1080x1080')
    vid_size_dropdown = mo.ui.dropdown(
        {res: tuple(map(int,res.split('x'))) for res in VID_SIZES},
        value='512x512', label='video size'
    )
    frame_rate_dropdown = mo.ui.dropdown({str(r): r for r in (20, 24, 25, 30)}, value='30', label='frame rate')
    return VID_SIZES, frame_rate_dropdown, vid_size_dropdown


@app.cell
def _(
    a_cen_slider,
    ab_r_slider,
    audio_button,
    b_cen_slider,
    c_cen_slider,
    cd_r_slider,
    d_cen_slider,
    frame_rate_dropdown,
    np,
    os,
    render_video,
    seq_box,
    track_duration,
    vid_seq_bar,
    vid_size_dropdown,
):
    def build_ui_video(_):
        audio_name = '{}.wav'.format(seq_box.value)
        video_name = '{}.mp4'.format(seq_box.value)
        if len(audio_button.value) > 1:
            with open(audio_name, 'wb') as f:
                f.write(audio_button.value)
        else:
            return None

        n_frames = np.rint(track_duration() / 44100 * frame_rate_dropdown.value).astype(int)
        width, height = vid_size_dropdown.value

        vid_frames = vid_seq_bar(
            seq_box.value,
            a_cen_slider.value, b_cen_slider.value, ab_r_slider.value,
            c_cen_slider.value, d_cen_slider.value, cd_r_slider.value,
            n_frames, width, height
        )

        if os.path.isfile(video_name):
            os.remove(video_name)

        render_video(audio_name, video_name, vid_frames, frame_rate_dropdown.value)

        return video_name
    return (build_ui_video,)


@app.cell
def _(mo, os):
    def display_video(vid_name):
        if os.path.isfile(vid_name):
            src = '/'.join(['file:/', os.getcwd(), vid_name])
            return mo.download(open(vid_name, 'rb'), filename=vid_name, mimetype='video/mp4')
        else:
            return ''
    return (display_video,)


@app.cell
def _(build_ui_video, mo):
    video_button = mo.ui.button(value='', on_click=build_ui_video, label='generate video')
    return (video_button,)


@app.cell
def _(sp, sys):
    if sys.platform != 'emscripten':
        ON_WASM = False
        try:
            GOT_FFMPEG = sp.run(['ffmpeg', '-version'], stdout=sp.DEVNULL).returncode == 0
        except:
            GOT_FFMPEG = False
    else:
        ON_WASM = True
        GOT_FFMPEG = False
    return GOT_FFMPEG, ON_WASM


@app.cell
def _(
    GOT_FFMPEG,
    ON_WASM,
    audio_button,
    display_video,
    frame_rate_dropdown,
    mo,
    vid_size_dropdown,
    video_button,
):
    def video_ui():
        if ON_WASM:
            return mo.center("(Running in WASM, can't build videos.)")
        elif not GOT_FFMPEG:
            return mo.center("(FFmpeg not found, can't build videos.)")
        elif len(audio_button.value) == 1:
            return mo.center('(No audio loop yet, generate one in order to build videos.)')
        else:
            return mo.vstack([
                mo.hstack([vid_size_dropdown, frame_rate_dropdown, video_button]),
                display_video(video_button.value)
            ], align='center')
    return (video_ui,)


@app.cell
def _(mo):
    mo.md("""*If* this is not being run via [WASM](https://docs.marimo.io/guides/wasm/) and [FFmpeg](https://www.ffmpeg.org/) is available, it can be used to generate a video to be combined with the audio loop. Each frame corresponds to one rotation of the $AB$ coefficients, so it wraps left to right, the sequence of frames corresponds to one rotation of the $CD$ coefficients, so it can loop endlessly.""")
    return


@app.cell
def _(video_ui):
    video_ui()
    return


@app.cell
def _(mo):
    mo.md("""In the default track, (as specifically not featured [here](https://kavarimusic.bandcamp.com/album/cult-tape-002-selected-ambient-works)) three bifurcation sequences are combined. Each uses the same centres ((3.75, 3.75) & (2.95, 2.95)) and radius (0.25) of circles, but there are subsitutions between the symbols in the coefficient sequences ($AACDCBB$, $CCABADD$ & $BBDCDAA$). The numbers of points in the $AB$ and $CD$ circles vary so they have the same product ((120, 24), (80, 36) & (60, 48)). Finally, the numbers of bins in the three sequences of histograms varies to give trebble, (1024) mids (512) and bass (256). 32768 samples are used per segment, with a hop-size of 8192.""")
    return


@app.cell
def _(
    bifurcation_noise,
    double_rot_coeffs,
    logistic_map,
    norm_samples,
    symbol_seq,
):
    def build_default_track():
        coeff = double_rot_coeffs(3.375, 3.375, 0.25, 120, 2.95, 2.95, 0.25, 24)
        seq = symbol_seq('AACDCBB', size=1024+128)
        hist = logistic_map(coeff, seq, 1024, 1024, 128)
        samples = 2.0 * norm_samples(bifurcation_noise(hist, 4 * 8192, 8, 0.75))

        coeff = double_rot_coeffs(3.375, 3.375, 0.25, 80, 2.95, 2.95, 0.25, 36)
        seq = symbol_seq('CCABADD', size=1024+128)
        hist = logistic_map(coeff, seq, 512, 1024, 128)
        samples += 1.5 * norm_samples(bifurcation_noise(hist, 4 * 8192, 8, 0.75))

        coeff = double_rot_coeffs(3.375, 3.375, 0.25, 60, 2.95, 2.95, 0.25, 48)
        seq = symbol_seq('BBDCDAA', size=1024+128)
        hist = logistic_map(coeff, seq, 256, 1024, 128)
        samples += 1.0 * norm_samples(bifurcation_noise(hist, 4 * 8192, 8, 0.75))

        return norm_samples(samples)
    return (build_default_track,)


@app.cell
def _(GOT_FFMPEG, build_default_track, np, render_video, vid_seq, wavfile):
    def build_default_vid():
        print('Generating the audio loop...')
        tk = build_default_track()
        wavfile.write('logistical_nightmares.wav', 44100, tk)
        print('Wrote to "logistical_nightmares.wav".')
        if GOT_FFMPEG:
            n_frames = int(np.rint(tk.shape[0] / 44100 * 30))
            vid_frames = vid_seq(
                'BBDCDAA',
                3.375, 3.375, 0.25,
                2.95, 2.95, 0.25,
                n_frames, 1080, 1080
            )
            print('Generating the video...')
            render_video(
                'logistical_nightmares.wav', 'logistical_nightmares.mp4', vid_frames, 30
            )
            print('Wrote to "logistical_nightmares.mp4".')
        else:
            print("No FFmpeg, can't generate the video.")
    return (build_default_vid,)


@app.cell
def _(build_default_vid, mo):
    if not mo.running_in_notebook():
        build_default_vid()
    return


if __name__ == "__main__":
    app.run()
