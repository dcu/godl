package main

import (
	"flag"
	"log"
	"os"

	"github.com/dcu/deepzen/image"
	"github.com/dcu/deepzen/imagenet"
	"github.com/dcu/deepzen/vgg"
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

	img, err := image.Load(flag.Args()[0], image.LoadOpts{
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
