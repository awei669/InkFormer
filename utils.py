import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import cm

'''please transfer the strokes to format 3 or format 5 before doing any processing'''

def trans2format3(strokes):
    '''Transform the strokes to format 3:
    (x, y, p) for every point
    - x and y are the coordinates of the point
    - p is the pen state, =-1 if the pen will lift, =1 if the pen is down
    '''
    format3 = []
    for stroke in strokes:
        for i in range(len(stroke)-1):
            format3.append((stroke[i][0], stroke[i][1], 1))
        format3.append((stroke[-1][0], stroke[-1][1], -1))
    format3 = np.stack(format3).astype(np.float32)
    return format3

def trans2format5(strokes):
    '''Transform the strokes to format 5:
    (x, y, p1, p2, p3) for every point
    - x and y are the coordinates of the point
    - p1, p2, p3 are the pen states, (1,0,0) if the pen is down, (0,1,0) if the pen will lift, (0,0,1) if the drawing is over
    '''
    format5 = []
    for stroke in strokes:
        for i in range(len(stroke)-1):
            format5.append((stroke[i][0], stroke[i][1], 1, 0, 0))
        format5.append((stroke[-1][0], stroke[-1][1], 0, 1, 0))
    format5[-1] = (format5[-1][0], format5[-1][1], 0, 0, 1)
    format5 = np.stack(format5).astype(np.float32)
    return format5

def format_switch(char, from_to='35'):
    '''switch the format of the character.
    char: np.array, shape=(L, 3)
    from_to: str, the format switch, e.g., '35' means from format 3 to format 5.
    '''
    if from_to == '35':
        xy = char[:, :2]
        ps3 = char[:, 2:]
        ps5 = []
        for p in ps3:
            if p == -1:
                ps5.append([0,1,0])
            else:
                ps5.append([1,0,0])
        ps5[-1] = [0,0,1]
        ps5 = np.stack(ps5, axis=0)
        new_char = np.concatenate([xy, ps5], axis=-1)
    elif from_to == '53':
        # TODO
        pass
    else:
        raise ValueError(f'unsupported format switch: {from_to}.')
    return new_char

def normalize_pts(points):
    '''Normalize x&y of every points to [0, 1]. 
    points should be shape of (n, 3) or (n, 5)'''
    xy = points[:,:2]
    ps = points[:,2:]
    min_x, min_y = np.min(xy, axis=0) ; max_x, max_y = np.max(xy, axis=0)
    mid_x, mid_y = (max_x+min_x)/2, (max_y+min_y)/2
    scale = max(max_x-min_x, max_y-min_y)   # 为了保持长宽比，取最大值
    xy = (xy - np.array([mid_x, mid_y])[None,:]) / scale    # normalize to [-0.5, 0.5]
    xy = xy + 0.5   # normalize to [0, 1]
    points = np.concatenate([xy, ps], axis=1)
    return points

def rmv_pads(points, format=3):
    '''Remove the pads in the strokes.
    points should be shape of (n, 3) or (n, 5)'''
    if format == 3:
        ps = points[:,2]
        sum_ps = np.convolve(ps, np.array([1, -1]), mode='same')
        eos = np.where(sum_ps == -2)[0]     # cuz only -1-1=-2
        if len(eos) > 0:
            last_valid = eos[-1]          # find the last -1-1=-2
        else:
            # 有些脏数据可能检测不到eos，这时候就直接返回原数据
            return points
        points = points[:last_valid+1]
    elif format == 5:
        ps = points[:,2:5]
        last_valid = np.where(ps[:,-1] == 1)[0]
        if len(last_valid) > 0:
            last_valid = last_valid[0]
        else:
            return points
        points = points[:last_valid+1]
    return points


def x2v(points):
    '''convert the absolute coordinates to the velocities.
    points: (L, 3or5)
    '''
    xy = points[:, :2] ; ps = points[:, 2:]
    dxdy = np.diff(xy, axis=0, prepend=np.zeros((1,2)))
    new_points = np.concatenate([dxdy, ps], axis=1)
    return new_points

def v2x(points):
    '''convert the velocities to the absolute coordinates.
    points: (L, 3or5)
    '''
    dxdy = points[:, :2] ; ps = points[:, 2:]
    xy = np.cumsum(dxdy, axis=0)
    new_points = np.concatenate([xy, ps], axis=1)
    return new_points
    


def draw_n_save(points, save_path=None, 
                canvas_size=256, line_width=2, point_radius=2, 
                color=False, draw_pads=False
                ):
    '''Draw the strokes and save the image. 
    points should be shape of (n, 3) or (n, 5)
    '''
    if isinstance(points, torch.Tensor):
        if points.is_cuda: points = points.cpu().detach()
        points = points.numpy()
    #
    format = points.shape[1]
    points[:,-1] = np.sign(points[:,-1])
    
    # rmv pads if needed
    if not draw_pads: points = rmv_pads(points, format)

    points = normalize_pts(points)
    
    # ready the canvas
    image = Image.new('RGB', (canvas_size, canvas_size), 'white')
    draw = ImageDraw.Draw(image)
    # ready the palette
    if color:
        color_map = cm.get_cmap('winter', 8)
        color_map = [tuple(int(255*x) for x in color_map(i)[:3]) for i in range(8)]
    else:
        color_map = ['black']
    color_idx = 0
    
    # let's draw
    for i in range(len(points)-1):
        if format == 3:
            x1, y1, p1 = points[i]
            x2, y2, p2 = points[i+1]
            draw_line_flag = (p1 == 1)
        elif format == 5:
            x1, y1, p11, p12, p13 = points[i]
            x2, y2, p21, p22, p23 = points[i+1]
            draw_line_flag = (p11 == 1)
        x1, y1, x2, y2 = x1*canvas_size, (1-y1)*canvas_size, x2*canvas_size, (1-y2)*canvas_size # vertical flip since the origin of PIL is at the top-left corner
        # draw the line
        if draw_line_flag:
            draw.line((x1, y1, x2, y2), width=line_width, fill=color_map[color_idx])
        else:
            color_idx = (color_idx+1) % len(color_map)
        # draw the points
        draw.ellipse((x1-point_radius, y1-point_radius, x1+point_radius, y1+point_radius), fill='red')
        draw.ellipse((x2-point_radius, y2-point_radius, x2+point_radius, y2+point_radius), fill='red')
    
    if save_path is not None:
        image.save(save_path)
        return
    else:
        return image




if __name__ == '__main__':
    # test_strokes draw a heart
    test_strokes = [[(0.0, 0.5), (-0.5, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, -1.0), 
                     (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), (0.0, 0.5)]]
    test_strokes = trans2format3(test_strokes)
    # test_strokes = trans2format5(test_strokes)
    test_strokes = normalize_pts(test_strokes)
    draw_n_save(test_strokes, 'test.png')