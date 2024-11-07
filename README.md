# CIFAR Image Classification Project

## Proje Hakkında

Bu proje, CIFAR-10 veri seti üzerinde görüntü sınıflandırma işlemi gerçekleştirmek amacıyla oluşturulmuştur. Projede, çeşitli derin öğrenme modelleri kullanılarak eğitim yapılmış ve farklı modellerin başarı oranları karşılaştırılmıştır.

## Veri Seti

CIFAR-10 veri seti, 10 farklı sınıftan oluşan toplamda 60,000 renklendirilmiş görüntü içerir. Sınıflar şunlardır:
- Uçak
- Otomobil
- Kuş
- Kedi
- Geyik
- Köpek
- Kurbağa
- At
- Gemi
- Kamyon

Her sınıfta 6,000 görüntü bulunur ve veri seti eğitim (50,000 görüntü) ve test (10,000 görüntü) olmak üzere iki kısıma ayrılmıştır.

## Kullanılan Modeller

Proje boyunca aşağıdaki modeller kullanılarak CIFAR-10 veri seti üzerinde eğitim yapılmıştır:

- Basit Convolutional Neural Network (CNN) modeli
- ResNet
- VGGNet
- AlexNet

Her modelin doğruluğu (accuracy) test verisi üzerinde değerlendirilmiştir.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python paketlerine ihtiyacınız var:

```bash
pip install torch torchvision matplotlib
