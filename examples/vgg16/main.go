package main

import (
	"flag"
	"log"
	"os"

	"github.com/dcu/godl/imagenet"
	"github.com/dcu/godl/imageutils"
	"github.com/dcu/godl/vgg"
	"github.com/fatih/color"
)

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	flag.Parse()
	if len(flag.Args()) == 0 {
		color.Yellow("pass an image to detect")

		os.Exit(1)
	}

	img, err := imageutils.Load(flag.Args()[0], imageutils.LoadOpts{
		TargetSize: []uint{224, 224},
	})
	handleErr(err)

	vgg16 := vgg.VGG16(vgg.Opts{
		PreTrained: true,
		Learnable:  false,
	})

	classifier := imagenet.NewClassifier(vgg16, 224, 224)

	label, prob, err := classifier.Predict(img)
	handleErr(err)

	log.Printf("%v: %.2f%%", label, prob*100)
}
