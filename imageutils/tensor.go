package imageutils

import (
	"fmt"
	"image"
	"image/color"
	"io/fs"
	"path/filepath"
	"regexp"

	"gorgonia.org/tensor"
)

var (
	imgRegexp = regexp.MustCompile(`\.(png|jpg|jpeg)$`)
)

type TensorMode string

const (
	TensorModeCaffe      TensorMode = "caffe"
	TensorModeTensorFlow TensorMode = "tensorflow"
	TensorModeTorch      TensorMode = "torch"
)

// ToTensorOpts are the options to convert the image to a tensor
type ToTensorOpts struct {
	// TensorMode is the mode to weight the pixels. The default one is Caffe
	TensorMode TensorMode
}

// ToTensorFromDirectory loads all images in a directory
func ToTensorFromDirectory(dirPath string, loadOpts LoadOpts, tensorOpts ToTensorOpts) (tensor.Tensor, error) {
	if len(loadOpts.TargetSize) != 2 {
		return nil, fmt.Errorf("TargetSize must be defined")
	}

	backing := []float64{}
	imagesCount := 0

	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !imgRegexp.MatchString(path) {
			return nil
		}

		img, err := Load(path, loadOpts)
		if err != nil {
			return err
		}

		weights := ToArray(img, tensorOpts)
		backing = append(backing, weights...)
		imagesCount++

		return nil
	})
	if err != nil {
		return nil, err
	}

	return tensor.New(
		tensor.Of(tensor.Float64),
		tensor.WithShape(imagesCount, 3, int(loadOpts.TargetSize[0]), int(loadOpts.TargetSize[1])), // count, channels, width, height
		tensor.WithBacking(backing),
	), nil
}

// ToTensor converts the given image to a tensor
func ToTensor(img image.Image, opts ToTensorOpts) tensor.Tensor {
	bounds := img.Bounds()

	return tensor.New(
		tensor.Of(tensor.Float64),
		tensor.WithShape(1, 3, bounds.Max.X, bounds.Max.Y), // batchSize, channels, width, height
		tensor.WithBacking(ToArray(img, opts)),
	)
}

// ToArray converts the image in a []float64
func ToArray(img image.Image, opts ToTensorOpts) []float64 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	pixels := make([]float64, 3*width*height)

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			w1, w2, w3 := pixelWeight(img.At(x, y), opts)

			pixels[width*y+x] = w1
			pixels[(width*y+x)+1*width*height] = w2
			pixels[(width*y+x)+2*width*height] = w3
		}
	}

	return pixels
}

func pixelWeight(pixel color.Color, opts ToTensorOpts) (float64, float64, float64) {
	r, g, b, _ := pixel.RGBA()

	switch opts.TensorMode {
	case TensorModeTensorFlow:
		// https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/imagenet_utils.py#L192

		return float64(r/256)/127.5 - 1.0,
			float64(g/256)/127.5 - 1.0,
			float64(b/256)/127.5 - 1.0
	case TensorModeTorch:
		// https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/imagenet_utils.py#L197
		mean := []float64{0.485, 0.456, 0.406}
		std := []float64{0.229, 0.224, 0.225}

		return (float64(r)/65536 - mean[0]) / std[0],
			(float64(g)/65536 - mean[1]) / std[1],
			(float64(b)/65536 - mean[2]) / std[2]
	default:
		// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/imagenet_utils.py#L202
		mean := []float64{103.939, 116.779, 123.68}

		// RGB -> BGR
		return float64(b/256) - mean[0],
			float64(g/256) - mean[1],
			float64(r/256) - mean[2]
	}
}
