package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
)

func main() {
	http.HandleFunc("/upload", uploadHandler)
	http.HandleFunc("/ocr", ocrHandler)
	log.Println("Start server")
	err := http.ListenAndServe("192.168.56.105:8085", nil)
	log.Println("Err", err)
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("upload")
	switch r.Method {
	case "POST":
		r.ParseMultipartForm(10 << 20) //10 MB
		file, handler, err := r.FormFile("file_image")
		if err != nil {
			log.Println("error retrieving file", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer file.Close()
		pathOCR := path.Join("pdf", handler.Filename)
		pathOCR = path.Join("pdf", "file.pdf")
		{
			dst, err := os.Create(pathOCR)
			if err != nil {
				log.Println("error creating file", err)
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer dst.Close()
			if _, err := io.Copy(dst, file); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}

		fmt.Fprintf(w, "uploaded file")
	}
}

func SaveFile() {

}

func ocrHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("ocr")
	switch r.Method {
	case "GET":
		absbat, _ := filepath.Abs("./ocr.bat")
		log.Println("abs", absbat)
		absocr, _ := filepath.Abs("./pdf/result.xlsx")
		log.Println("abs", absocr)

		if err := exec.Command(absbat).Run(); err != nil {
			log.Println("Error run bat file", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		http.ServeFile(w, r, absocr)

		//fmt.Fprintf(w, "uploaded file")
	}
}
