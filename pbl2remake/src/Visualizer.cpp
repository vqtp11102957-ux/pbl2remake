#include "Visualizer.h"

#include <algorithm>
#include <cmath>

Image Visualizer::toColor(const Image& img) {
    if (img.isEmpty()) {
        return Image();
    }
    if (img.getChannels() == 3) {
        return img;
    }
    Image color(img.getWidth(), img.getHeight(), 3);
    color.paste(img, 0, 0);
    return color;
}

void Visualizer::drawOutline(Image& img, const Rect& r, int red, int green, int blue, int thickness) {
    if (img.isEmpty() || r.w <= 0 || r.h <= 0 || thickness < 1) {
        return;
    }
    unsigned char R = static_cast<unsigned char>(red);
    unsigned char G = static_cast<unsigned char>(green);
    unsigned char B = static_cast<unsigned char>(blue);
    int t = std::min(thickness, std::min(r.w, r.h));
    img.fillRect(Rect(r.x, r.y, r.w, t), R, G, B);                  // cạnh trên
    img.fillRect(Rect(r.x, r.y + r.h - t, r.w, t), R, G, B);        // cạnh dưới
    img.fillRect(Rect(r.x, r.y, t, r.h), R, G, B);                  // cạnh trái
    img.fillRect(Rect(r.x + r.w - t, r.y, t, r.h), R, G, B);        // cạnh phải
}

void Visualizer::tint(Image& img, const Rect& r, int red, int green, int blue, int percent) {
    if (img.isEmpty() || img.getChannels() != 3) {
        return;
    }
    int x0 = std::max(0, r.x);
    int y0 = std::max(0, r.y);
    int x1 = std::min(img.getWidth(), r.x + r.w);
    int y1 = std::min(img.getHeight(), r.y + r.h);
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int oldR = img.getPixel(x, y, 0);
            int oldG = img.getPixel(x, y, 1);
            int oldB = img.getPixel(x, y, 2);
            img.setPixel(x, y, 0, static_cast<unsigned char>((oldR * (100 - percent) + red * percent) / 100));
            img.setPixel(x, y, 1, static_cast<unsigned char>((oldG * (100 - percent) + green * percent) / 100));
            img.setPixel(x, y, 2, static_cast<unsigned char>((oldB * (100 - percent) + blue * percent) / 100));
        }
    }
}

Image Visualizer::fitInto(const Image& img, int areaW, int areaH, FitInfo* info) {
    Image canvas(areaW, areaH, 3, 30);
    FitInfo fit;
    if (!img.isEmpty() && areaW > 0 && areaH > 0) {
        double scale = std::min(static_cast<double>(areaW) / img.getWidth(),
                                static_cast<double>(areaH) / img.getHeight());
        int drawW = std::max(1, static_cast<int>(std::floor(img.getWidth() * scale)));
        int drawH = std::max(1, static_cast<int>(std::floor(img.getHeight() * scale)));
        Image scaled = (drawW == img.getWidth() && drawH == img.getHeight()) ? img : img.resized(drawW, drawH);
        fit.scale = scale;
        fit.drawWidth = drawW;
        fit.drawHeight = drawH;
        fit.offsetX = (areaW - drawW) / 2;
        fit.offsetY = (areaH - drawH) / 2;
        canvas.paste(scaled, fit.offsetX, fit.offsetY);
    }
    if (info != 0) {
        *info = fit;
    }
    return canvas;
}

Rect Visualizer::mapRect(const Rect& r, const FitInfo& fit) {
    int x0 = static_cast<int>(std::floor(r.x * fit.scale + 0.5)) + fit.offsetX;
    int y0 = static_cast<int>(std::floor(r.y * fit.scale + 0.5)) + fit.offsetY;
    int x1 = static_cast<int>(std::floor((r.x + r.w) * fit.scale + 0.5)) + fit.offsetX;
    int y1 = static_cast<int>(std::floor((r.y + r.h) * fit.scale + 0.5)) + fit.offsetY;
    return Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

void Visualizer::viewToImage(const FitInfo& fit, int viewX, int viewY, int& imageX, int& imageY) {
    double scale = (fit.scale > 0.0) ? fit.scale : 1.0;
    imageX = static_cast<int>(std::floor((viewX - fit.offsetX) / scale));
    imageY = static_cast<int>(std::floor((viewY - fit.offsetY) / scale));
}

Image Visualizer::renderBoxes(const Image& img, const std::vector<Rect>& boxes,
                              int areaW, int areaH, int red, int green, int blue) {
    FitInfo fit;
    Image canvas = fitInto(img, areaW, areaH, &fit);
    for (size_t i = 0; i < boxes.size(); i++) {
        drawOutline(canvas, mapRect(boxes[i], fit), red, green, blue, 2);
    }
    return canvas;
}

Image Visualizer::renderFeature(const Image& img, const HaarFeature& feature,
                                int areaW, int areaH, FitInfo* info) {
    FitInfo fit;
    Image canvas = fitInto(img, areaW, areaH, &fit);
    std::vector<Rect> whites = feature.getWhiteRects();
    std::vector<Rect> blacks = feature.getBlackRects();
    for (size_t i = 0; i < whites.size(); i++) {
        Rect view = mapRect(whites[i], fit);
        tint(canvas, view, 0, 220, 0, 40);
        drawOutline(canvas, view, 0, 255, 0, 2);
    }
    for (size_t i = 0; i < blacks.size(); i++) {
        Rect view = mapRect(blacks[i], fit);
        tint(canvas, view, 230, 0, 0, 40);
        drawOutline(canvas, view, 255, 0, 0, 2);
    }
    if (info != 0) {
        *info = fit;
    }
    return canvas;
}
