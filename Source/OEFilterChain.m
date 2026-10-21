// Copyright (c) 2019, OpenEmu Team
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the OpenEmu Team nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY OpenEmu Team ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OpenEmu Team BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@import MetalKit;

#import "OEFilterChain.h"
#import "OEShaderPassCompiler.h"
#import "OELogging.h"
#import "OEPixelBuffer+Internal.h"
#import "RendererCommon.h"
#import <OpenEmuShaders/OpenEmuShaders.h>
#import <CoreImage/CoreImage.h>
#import <OpenEmuShaders/OpenEmuShaders-Swift.h>
#include <stdlib.h>

static const char *OEFilterChainScaleName(OEShaderPassScale scale)
{
    switch (scale) {
        case OEShaderPassScaleSource:
            return "source";
        case OEShaderPassScaleViewport:
            return "viewport";
        case OEShaderPassScaleAbsolute:
            return "absolute";
        default:
            return "unspecified";
    }
}

// BOXER-FORK: NSLog mirror for shader diagnostics; safe per Phase 1.
static void BXShaderDiagLog(NSString *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    NSString *msg = [[NSString alloc] initWithFormat:fmt arguments:args];
    va_end(args);
    NSLog(@"[BoxerShaderDiag] %@", msg);
}

static NSError *OEFilterChainShaderError(NSString *message)
{
    return [NSError errorWithDomain:@"org.openemu.OpenEmuShaders.ShaderLoading"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

// BOXER-FORK: format names for shader diagnostics.
static NSString *BXPixelFormatName(MTLPixelFormat format)
{
    switch (format) {
        case MTLPixelFormatInvalid:
            return @"Invalid";
        case MTLPixelFormatBGRA8Unorm:
            return @"BGRA8Unorm";
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return @"BGRA8Unorm_sRGB";
        case MTLPixelFormatRGBA8Unorm:
            return @"RGBA8Unorm";
        case MTLPixelFormatRGBA8Unorm_sRGB:
            return @"RGBA8Unorm_sRGB";
        case MTLPixelFormatRGBA16Float:
            return @"RGBA16Float";
        default:
            return [NSString stringWithFormat:@"MTLPixelFormat(%lu)", (unsigned long)format];
    }
}

#define MTLALIGN(x) __attribute__((aligned(x)))

typedef struct
{
    float x;
    float y;
    float z;
    float w;
} float4_t;

typedef struct texture
{
    id<MTLTexture> view;
    float4_t       viewSize;
} texture_t;

@implementation OEFilterChain
{
    id<MTLDevice>       _device;
    id<MTLLibrary>      _library;
    MTKTextureLoader    *_loader;
    MTLPixelConverter   *_converter;
    Vertex              _vertex[4];
    Vertex              _vertexFlipped[4];
    OEPixelBuffer       *_pixelBuffer;
    
    id<MTLSamplerState> _samplers[OEShaderPassFilterCount][OEShaderPassWrapCount];
    
    // #pragma mark screenshots
    id<MTLCommandQueue> _commandQueue;
    id<MTLTexture>      _screenshotTexture;
    CIContext           *_ciContext;
    
    SlangShader *_shader;
    
    NSUInteger _frameCount;
    NSUInteger _passCount;
    NSUInteger _lastPassIndex;
    NSUInteger _lutCount;
    NSUInteger _historyCount;
    
    id<MTLTexture> _texture;       // final render texture
    texture_t      _sourceTextures[kMaxFrameHistory + 1];
    
    struct
    {
        MTLViewport viewport;
        float4_t    outputSize;
    }              _outputFrame;
    
    struct
    {
        id<MTLBuffer>              buffers[kMaxConstantBuffers];
        id<MTLBuffer>              vBuffers[kMaxConstantBuffers]; // array used for vertex binding
        id<MTLBuffer>              fBuffers[kMaxConstantBuffers]; // array used for fragment binding
        texture_t                  renderTarget;
        texture_t                  feedbackTarget;
        uint32_t                   frameCount;
        uint32_t                   frameCountMod;
        int32_t                    frameDirection;
        ShaderPassBindings         *bindings;
        MTLViewport                viewport;
        id<MTLRenderPipelineState> state;
        BOOL                       hasFeedback;
    }              _pass[kMaxShaderPasses];
    
    texture_t _luts[kMaxTextures];
    BOOL      _lutsFlipped;
    
    BOOL          _renderTargetsNeedResize;
    BOOL          _historyNeedsInit;
    
    BOOL          _usesCustomOutputViewport;
    MTLViewport   _customOutputViewport;
    
    CGRect _sourceRect;
    CGSize _aspectSize;
    CGSize _drawableSize;
    // aspect-adjusted output bound
    NSRect _outputBounds;
    
    // render target layer state
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLSamplerState>        _samplerStateLinear;
    id<MTLSamplerState>        _samplerStateNearest;
    unsigned                   _rotation;
    Uniforms                   _uniforms;
    Uniforms                   _uniformsNoRotate;
    id<MTLTexture>             _checkers;
    
    // device behavior
    BOOL _deviceUnifiedMemory;
    BOOL _diagnosticPending;

}

- (instancetype)initWithDevice:(id<MTLDevice>)device
{
    self = [super init];
    
    NSError *err = nil;
    _device    = device;
    
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_15
    // gather capabilities
    if (@available(macOS 10.15, ios 11, *)) {
        _deviceUnifiedMemory = _device.hasUnifiedMemory;
    } else {
        _deviceUnifiedMemory = NO;
    }
#else
    _deviceUnifiedMemory = NO;
#endif
    
    _library   = [_device newDefaultLibraryWithBundle:[NSBundle bundleForClass:self.class] error:&err];
    if (err != nil) {
        os_log_error(OE_LOG_DEFAULT, "error initializing Metal library: %{public}@", err.localizedDescription);
    }
    assert(err == nil);
    _loader    = [[MTKTextureLoader alloc] initWithDevice:device];
    _converter = [[MTLPixelConverter alloc] initWithDevice:_device
                                                     library:_library
                                                       error:&err];
    if (err != nil) {
        os_log_error(OE_LOG_DEFAULT, "error initializing pixel converter: %{public}@", err.localizedDescription);
    }
    assert(err == nil);
    
    if (![self OE_initState]) {
        return nil;
    }
    [self OE_initSamplers];
    
    memcpy(_vertex, (void *)(Vertex[4]){
        {simd_make_float4(0, 1, 0, 1), simd_make_float2(0, 1)},
        {simd_make_float4(1, 1, 0, 1), simd_make_float2(1, 1)},
        {simd_make_float4(0, 0, 0, 1), simd_make_float2(0, 0)},
        {simd_make_float4(1, 0, 0, 1), simd_make_float2(1, 0)},
    }, sizeof(_vertex));
    
    memcpy(_vertexFlipped, (void *)(Vertex[4]) {
        {simd_make_float4(0, 1, 0, 1), simd_make_float2(0, 0)},
        {simd_make_float4(0, 0, 0, 1), simd_make_float2(0, 1)},
        {simd_make_float4(1, 1, 0, 1), simd_make_float2(1, 0)},
        {simd_make_float4(1, 0, 0, 1), simd_make_float2(1, 1)},
    }, sizeof(_vertexFlipped));
    
    [self setRotation:0];
    
    _renderTargetsNeedResize = YES;
    _frameDirection          = 1;
    
    return self;
}

- (BOOL)OE_initState
{
    {
        MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
        vd.attributes[0].offset = offsetof(Vertex, position);
        vd.attributes[0].format = MTLVertexFormatFloat4;
        vd.attributes[0].bufferIndex = BufferIndexPositions;
        vd.attributes[1].offset = offsetof(Vertex, texCoord);
        vd.attributes[1].format = MTLVertexFormatFloat2;
        vd.attributes[1].bufferIndex = BufferIndexPositions;
        vd.layouts[4].stride         = sizeof(Vertex);
        vd.layouts[4].stepFunction   = MTLVertexStepFunctionPerVertex;
        
        MTLRenderPipelineDescriptor *psd = [MTLRenderPipelineDescriptor new];
        psd.label = @"Pipeline+No Alpha";
        
        MTLRenderPipelineColorAttachmentDescriptor *ca = psd.colorAttachments[0];
        ca.pixelFormat                 = MTLPixelFormatBGRA8Unorm; // NOTE(sgc): expected layer format (could be taken from layer.pixelFormat)
        ca.blendingEnabled             = NO;
        ca.sourceAlphaBlendFactor      = MTLBlendFactorSourceAlpha;
        ca.sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
        ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        ca.destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
        
        psd.sampleCount      = 1;
        psd.vertexDescriptor = vd;
        psd.vertexFunction   = [_library newFunctionWithName:@"basic_vertex_proj_tex"];
        psd.fragmentFunction = [_library newFunctionWithName:@"basic_fragment_proj_tex"];
        
        NSError *err;
        _pipelineState = [_device newRenderPipelineStateWithDescriptor:psd error:&err];
        if (err != nil) {
            os_log_error(OE_LOG_DEFAULT, "error creating pipeline state: %{public}@", err.localizedDescription);
            return NO;
        }
    }
    
    {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        _samplerStateNearest = [_device newSamplerStateWithDescriptor:sd];
        
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        _samplerStateLinear = [_device newSamplerStateWithDescriptor:sd];
    }
    
    return YES;
}

- (void)OE_initSamplers
{
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    
    /* Initialize samplers */
    for (unsigned i = 0; i < OEShaderPassWrapCount; i++) {
        NSString *label = nil;
        switch (i) {
            case OEShaderPassWrapBorder:
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_15
                if (@available(macOS 10.15, iOS 11, *)) {
                    if ([_device supportsFamily:MTLGPUFamilyApple1]) {
                        label = @"clamp_to_zero";
                        sd.sAddressMode = MTLSamplerAddressModeClampToZero;
                        break;
                    }
                }
#endif
                label = @"clamp_to_border";
                sd.sAddressMode = MTLSamplerAddressModeClampToBorderColor;
                break;
            
            case OEShaderPassWrapEdge:
                label = @"clamp_to_edge";
                sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
                break;
            
            case OEShaderPassWrapRepeat:
                label = @"repeat";
                sd.sAddressMode = MTLSamplerAddressModeRepeat;
                break;
            
            case OEShaderPassWrapMirroredRepeat:
                label = @"mirrored_repeat";
                sd.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
                break;
            
            default:
                continue;
        }
        sd.tAddressMode = sd.sAddressMode;
        sd.rAddressMode = sd.sAddressMode;
        sd.minFilter    = MTLSamplerMinMagFilterLinear;
        sd.magFilter    = MTLSamplerMinMagFilterLinear;
        sd.label = [NSString stringWithFormat:@"%@ (linear)", label];
        
        id<MTLSamplerState> ss = [_device newSamplerStateWithDescriptor:sd];
        _samplers[OEShaderPassFilterLinear][i] = ss;
        
        sd.minFilter = MTLSamplerMinMagFilterNearest;
        sd.magFilter = MTLSamplerMinMagFilterNearest;
        sd.label = [NSString stringWithFormat:@"%@", label];
        
        ss = [_device newSamplerStateWithDescriptor:sd];
        _samplers[OEShaderPassFilterNearest][i] = ss;
    }
}

- (void)setRotation:(unsigned)rotation
{
    _rotation = 270 * rotation;
    
    /* Calculate projection. */
    _uniformsNoRotate.projectionMatrix = matrix_proj_ortho(0, 1, 0, 1);
    
    simd_float4x4 rot = matrix_rotate_z((float)(M_PI * _rotation / 180.0f));
    _uniforms.projectionMatrix = simd_mul(rot, _uniformsNoRotate.projectionMatrix);
}

- (void)setDefaultFilteringLinear:(BOOL)linear
{
    for (int i = 0; i < OEShaderPassWrapCount; i++) {
        if (linear)
            _samplers[OEShaderPassFilterUnspecified][i] = _samplers[OEShaderPassFilterLinear][i];
        else
            _samplers[OEShaderPassFilterUnspecified][i] = _samplers[OEShaderPassFilterNearest][i];
    }
}

- (void)setSourceTexture:(id<MTLTexture>)sourceTexture {
    if (_sourceTexture == sourceTexture) {
        return;
    }
    
    _pixelBuffer = nil;
    _texture = nil;
    _sourceTexture = sourceTexture;
}

- (void)setSourceTextureIsFlipped:(BOOL)sourceTextureIsFlipped {
    _sourceTextureIsFlipped = sourceTextureIsFlipped;
}

- (void)setFrameDirection:(NSInteger)frameDirection {
    frameDirection = MAX(MIN(1, frameDirection), -1);
    if (frameDirection == _frameDirection) {
        return;
    }
    
    [self willChangeValueForKey:@"frameDirection"];
    _frameDirection = frameDirection;
    [self didChangeValueForKey:@"frameDirection"];
}

- (void)OE_updateHistory
{
    if (_shader) {
        if (_historyCount) {
            if (_historyNeedsInit) {
                [self OE_initHistory];
            } else {
                texture_t       tmp = _sourceTextures[_historyCount];
                for (NSUInteger k   = _historyCount; k > 0; k--) {
                    _sourceTextures[k] = _sourceTextures[k - 1];
                }
                _sourceTextures[0]  = tmp;
            }
        }
    }
    
    if (_historyCount == 0 && _sourceTexture) {
        [self OE_initTexture:&_sourceTextures[0] withTexture:_sourceTexture];
        return;
    }
    
    /* either no history, or we moved a texture of a different size in the front slot */
    if (_sourceTextures[0].viewSize.x != _sourceRect.size.width || _sourceTextures[0].viewSize.y != _sourceRect.size.height) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:_sourceRect.size.width
                                                                                     height:_sourceRect.size.height
                                                                                  mipmapped:NO];
        td.storageMode  = MTLStorageModePrivate;
        td.usage        = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        [self OE_initTexture:&_sourceTextures[0] withDescriptor:td];
    }
}

/*
 * Take the raw visible game rect and turn it into a smaller rect
 * which is centered inside 'bounds' and has aspect ratio 'aspectSize'.
 * ATM we try to fill the window, but maybe someday we'll support fixed zooms.
 */
static NSRect FitAspectRectIntoRect(CGSize aspectSize, CGSize size)
{
    CGFloat wantAspect = aspectSize.width / aspectSize.height;
    CGFloat viewAspect = size.width / size.height;
    
    CGFloat minFactor;
    NSRect  outRect;
    
    if (viewAspect >= wantAspect) {
        // Raw image is too wide (normal case), squish inwards
        minFactor = wantAspect / viewAspect;
        
        outRect.size.height = size.height;
        outRect.size.width  = size.width * minFactor;
    } else {
        // Raw image is too tall, squish upwards
        minFactor = viewAspect / wantAspect;
        
        outRect.size.height = size.height * minFactor;
        outRect.size.width  = size.width;
    }
    
    outRect.origin.x = (size.width - outRect.size.width) / 2;
    outRect.origin.y = (size.height - outRect.size.height) / 2;
    
    
    // This is going into a Nearest Neighbor, so the edges should be on pixels!
    return NSIntegralRectWithOptions(outRect, NSAlignAllEdgesNearest);
}

- (void)OE_resize {
    NSRect bounds = FitAspectRectIntoRect(_aspectSize, _drawableSize);
    if (CGPointEqualToPoint(_outputBounds.origin, bounds.origin) && CGSizeEqualToSize(_outputBounds.size, bounds.size) && !_usesCustomOutputViewport) {
        return;
    }
    
    _outputBounds = bounds;
    CGSize size = _outputBounds.size;
    
    _outputFrame.viewport = (MTLViewport){
        .originX = _outputBounds.origin.x,
        .originY = _outputBounds.origin.y,
        .width   = size.width,
        .height  = size.height,
        .znear   = 0.0,
        .zfar    = 1.0,
    };
    _outputFrame.outputSize.x     = size.width;
    _outputFrame.outputSize.y     = size.height;
    _outputFrame.outputSize.z     = 1.0f / size.width;
    _outputFrame.outputSize.w     = 1.0f / size.height;
    
    if (_usesCustomOutputViewport) {
        _outputFrame.viewport = _customOutputViewport;
        _outputFrame.outputSize.x = _customOutputViewport.width;
        _outputFrame.outputSize.y = _customOutputViewport.height;
        _outputFrame.outputSize.z = _customOutputViewport.width > 0.0 ? 1.0f / _customOutputViewport.width : 0.0f;
        _outputFrame.outputSize.w = _customOutputViewport.height > 0.0 ? 1.0f / _customOutputViewport.height : 0.0f;
    }
    
    if (_shader) {
        _renderTargetsNeedResize = YES;
    }
}

- (void)setSourceRect:(CGRect)rect aspect:(CGSize)aspect
{
    if (CGRectEqualToRect(_sourceRect, rect) && CGSizeEqualToSize(_aspectSize, aspect)) {
        return;
    }
    
    _sourceRect = rect;
    if (_pixelBuffer) {
        _pixelBuffer.outputRect = rect;
    }
    _aspectSize = aspect;
    [self OE_resize];
}

- (void)setDrawableSize:(CGSize)size {
    if (CGSizeEqualToSize(size, _drawableSize)) {
        return;
    }
    
    _drawableSize = size;
    _diagnosticPending = YES;

    [self OE_resize];
}
- (void)setOutputViewport:(MTLViewport)viewport
{
    _usesCustomOutputViewport = YES;
    _customOutputViewport = viewport;
    _outputFrame.viewport = viewport;
    _outputFrame.outputSize.x = viewport.width;
    _outputFrame.outputSize.y = viewport.height;
    _outputFrame.outputSize.z = viewport.width > 0.0 ? 1.0f / viewport.width : 0.0f;
    _outputFrame.outputSize.w = viewport.height > 0.0 ? 1.0f / viewport.height : 0.0f;
    _renderTargetsNeedResize = YES;
}

- (void)resetOutputViewport
{
    if (!_usesCustomOutputViewport) {
        return;
    }
    _usesCustomOutputViewport = NO;
    _outputBounds = NSZeroRect;
    [self OE_resize];
}


- (OEPixelBuffer *)newBufferWithFormat:(OEMTLPixelFormat)format height:(NSUInteger)height bytesPerRow:(NSUInteger)bytesPerRow
{
    _pixelBuffer = [OEPixelBuffer bufferWithDevice:_device converter:_converter
                                            format:format
                                            height:height bytesPerRow:bytesPerRow];
    _pixelBuffer.outputRect = _sourceRect;
    
    return _pixelBuffer;
}

- (OEPixelBuffer *)newBufferWithFormat:(OEMTLPixelFormat)format height:(NSUInteger)height bytesPerRow:(NSUInteger)bytesPerRow bytes:(void *)pointer
{
    _pixelBuffer = [OEPixelBuffer bufferWithDevice:_device converter:_converter
                                            format:format
                                            height:height bytesPerRow:bytesPerRow
                                             bytes:pointer];
    _pixelBuffer.outputRect = _sourceRect;
    
    return _pixelBuffer;
}

- (id<MTLCommandQueue>)commandQueue
{
    if (_commandQueue == nil) {
        _commandQueue = [_device newCommandQueue];
    }
    return _commandQueue;
}

- (CIContext *)OE_ciContext
{
    if (_ciContext == nil) {
        _ciContext = [CIContext contextWithMTLDevice:_device];
    }
    return _ciContext;
}

- (id<MTLTexture>)screenshotTexture
{
    if (_screenshotTexture == nil ||
            _screenshotTexture.width != _drawableSize.width ||
            _screenshotTexture.height != _drawableSize.height) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                      width:(NSUInteger)_drawableSize.width
                                                                                     height:(NSUInteger)_drawableSize.height
                                                                                  mipmapped:NO];
        td.storageMode  = MTLStorageModePrivate;
        td.usage        = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        _screenshotTexture = [_device newTextureWithDescriptor:td];
    }
    return _screenshotTexture;
}

- (NSBitmapImageRep *)blackImage {
    NSBitmapImageRep *img = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:nil
                                                                    pixelsWide:32
                                                                    pixelsHigh:32
                                                                 bitsPerSample:8
                                                               samplesPerPixel:4
                                                                      hasAlpha:YES
                                                                      isPlanar:NO
                                                                colorSpaceName:NSDeviceRGBColorSpace
                                                                   bytesPerRow:32 * 4
                                                                  bitsPerPixel:32];
    
    NSGraphicsContext *ctx = [NSGraphicsContext graphicsContextWithBitmapImageRep:img];
    [NSGraphicsContext saveGraphicsState];
    NSGraphicsContext.currentContext = ctx;
    [NSColor.blackColor drawSwatchInRect:NSMakeRect(0, 0, 32, 32)];
    [ctx flushGraphics];
    [NSGraphicsContext restoreGraphicsState];

    return img;
}

- (NSBitmapImageRep *)imageWithCIImage:(CIImage *)img {
    CGColorSpaceRef cs  = nil;
    CGImageRef cgImgTmp = nil;
    CGImageRef cgImg    = nil;
    @try {
        // TODO: use same color space as OEGameHelperMetalLayer
        cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);

        CIContext *ctx = [self OE_ciContext];
        // Specify the same color space as the original CIImage to preserve the original
        // pixel values of the MTLTexture
        cgImgTmp = [ctx createCGImage:img fromRect:img.extent format:kCIFormatBGRA8 colorSpace:img.colorSpace];
        // Override the original color space and set the correct one
        cgImg = CGImageCreateCopyWithColorSpace(cgImgTmp, cs);
        if (cgImg) {
            return [[NSBitmapImageRep alloc] initWithCGImage:cgImg];
        }
        os_log_error(OE_LOG_DEFAULT, "%{public}s: conversion failed, returning black image", __FUNCTION__);
        return [self blackImage];
    } @finally {
        if (cgImgTmp)   CGImageRelease(cgImgTmp);
        if (cgImg)      CGImageRelease(cgImg);
        if (cs)         CGColorSpaceRelease(cs);
    }
}

- (NSBitmapImageRep *)captureSourceImage
{
    NSDictionary<CIImageOption, id> *opts = @{
                                              kCIImageNearestSampling: @YES,
                                              };
    CIImage *img  = [[CIImage alloc] initWithMTLTexture:_texture options:opts];
    img = [img imageBySettingAlphaOneInExtent:img.extent];
    if (!_sourceTexture || !_sourceTextureIsFlipped) {
        img = [img imageByApplyingTransform:CGAffineTransformTranslate(CGAffineTransformMakeScale(1, -1), 0, img.extent.size.height)];
    }
    
    return [self imageWithCIImage:img];
}

- (NSBitmapImageRep *)captureOutputImage
{
    // render the filtered image
    id<MTLTexture>          tex  = [self screenshotTexture];
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor new];
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].texture    = tex;
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    [self renderWithCommandBuffer:commandBuffer renderPassDescriptor:rpd];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // copy to an NSBitmap
    NSDictionary<CIImageOption, id> *opts = @{
                                              kCIImageNearestSampling: @YES,
                                              };
    CIImage *img = [[CIImage alloc] initWithMTLTexture:tex options:opts];
    img = [img imageBySettingAlphaOneInExtent:img.extent];
    img = [img imageByCroppingToRect:CGRectMake(_outputBounds.origin.x, _outputBounds.origin.y, _outputBounds.size.width, _outputBounds.size.height)];
    img = [img imageByApplyingTransform:CGAffineTransformTranslate(CGAffineTransformMakeScale(1, -1), 0, img.extent.size.height)];
    
    return [self imageWithCIImage:img];
}

- (void)OE_prepareNextFrameWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    _frameCount++;
    [self OE_resizeRenderTargets];
    [self OE_updateHistory];
    _texture = _sourceTextures[0].view;
    
    if (_pixelBuffer) {
        [_pixelBuffer prepareWithCommandBuffer:commandBuffer texture:_texture];
        return;
    }
    
    if (_sourceTexture) {
        if (_historyCount == 0) {
            // _sourceTextures[0].view == _sourceTexture
            return;
        }
        
        MTLOrigin                 orig = {.x = _sourceRect.origin.x, .y = _sourceRect.origin.y, .z = 0};
        MTLSize                   size = {.width = (NSUInteger)_sourceRect.size.width, .height = (NSUInteger)_sourceRect.size.height, .depth = 1};
        MTLOrigin                 zero = {0};
        id<MTLBlitCommandEncoder> bce  = [commandBuffer blitCommandEncoder];
        [bce copyFromTexture:_sourceTexture sourceSlice:0 sourceLevel:0 sourceOrigin:orig sourceSize:size
                   toTexture:_texture destinationSlice:0 destinationLevel:0 destinationOrigin:zero];
        [bce endEncoding];
    }
}

- (void)OE_initTexture:(texture_t *)t withDescriptor:(MTLTextureDescriptor *)td
{
    t->view       = [_device newTextureWithDescriptor:td];
    t->viewSize.x = td.width;
    t->viewSize.y = td.height;
    t->viewSize.z = 1.0f / td.width;
    t->viewSize.w = 1.0f / td.height;
}

- (void)OE_initTexture:(texture_t *)t withTexture:(id<MTLTexture>)tex
{
    t->view       = tex;
    t->viewSize.x = tex.width;
    t->viewSize.y = tex.height;
    t->viewSize.z = 1.0f / tex.width;
    t->viewSize.w = 1.0f / tex.height;
}

- (void)OE_initHistory
{
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                  width:_sourceRect.size.width
                                                                                 height:_sourceRect.size.height
                                                                              mipmapped:NO];
    td.storageMode  = MTLStorageModePrivate;
    td.usage        = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    
    for (int i = 0; i < _historyCount + 1; i++) {
        [self OE_initTexture:&_sourceTextures[i] withDescriptor:td];
    }
    _historyNeedsInit = NO;
}

- (void)OE_renderTexture:(id<MTLTexture>)texture renderCommandEncoder:(id<MTLRenderCommandEncoder>)rce
{
    [rce setVertexBytes:&_uniforms length:sizeof(_uniforms) atIndex:BufferIndexUniforms];
    [rce setRenderPipelineState:_pipelineState];
    [rce setFragmentSamplerState:_samplerStateNearest atIndex:SamplerIndexDraw];
    [rce setViewport:_outputFrame.viewport];
    [rce setFragmentTexture:texture atIndex:TextureIndexColor];
    [rce drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

- (void)renderWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
           renderPassDescriptor:(MTLRenderPassDescriptor *)rpd
{
    [self renderOffscreenPassesWithCommandBuffer:commandBuffer];
    id<MTLRenderCommandEncoder> rce = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
    [self renderFinalPassWithCommandEncoder:rce];
    [rce endEncoding];
}

- (void)renderOffscreenPassesWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
{
    [self OE_prepareNextFrameWithCommandBuffer:commandBuffer];
    [self OE_logResolvedDimensionsIfNeeded];

    [self OE_updateBuffersForPasses];
    
    if (!_shader || _passCount == 0) {
        return;
    }
    
    // flip feedback render targets
    for (NSUInteger i = 0; i < _passCount; i++) {
        if (_pass[i].hasFeedback) {
            texture_t tmp = _pass[i].feedbackTarget;
            _pass[i].feedbackTarget = _pass[i].renderTarget;
            _pass[i].renderTarget   = tmp;
        }
    }
    
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor new];
    rpd.colorAttachments[0].loadAction  = MTLLoadActionDontCare;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    BOOL lastPassIsDirect = (_pass[_lastPassIndex].renderTarget.view == nil);
    NSUInteger count = _passCount - (lastPassIsDirect ? 1 : 0);
    
    for (NSUInteger i = 0; i < count; i++) {
        rpd.colorAttachments[0].texture = _pass[i].renderTarget.view;
        id<MTLRenderCommandEncoder> rce = [commandBuffer renderCommandEncoderWithDescriptor:rpd];
        [rce setVertexBytes:_vertex length:sizeof(_vertex) atIndex:BufferIndexPositions];
        [rce setViewport:_pass[i].viewport];
        [self OE_renderPassIndex:i renderCommandEncoder:rce];
        [rce endEncoding];
    }
}

- (void)OE_updateBuffersForPasses {
    for (int i = 0; i < _passCount; i++) {
        _pass[i].frameDirection = (int32_t)_frameDirection;
        _pass[i].frameCount     = (uint32_t)_frameCount;
        if (_pass[i].frameCountMod) {
            _pass[i].frameCount %= _pass[i].frameCountMod;
        }
        
        for (unsigned j = 0; j < kMaxConstantBuffers; j++) {
            ShaderPassBufferBinding *sem = _pass[i].bindings.buffers[j];
            
            if (sem.stageUsage && sem.uniforms.count > 0) {
                id<MTLBuffer> buffer = _pass[i].buffers[j];
                void          *data  = buffer.contents;
                
                for (ShaderPassUniformBinding *uniform in sem.uniforms) {
                    memcpy((uint8_t *)data + uniform.offset, uniform.data, uniform.size);
                }
                
                if (!_deviceUnifiedMemory) {
                    [buffer didModifyRange:NSMakeRange(0, buffer.length)];
                }
            }
        }
    }
}

- (void)OE_renderPassIndex:(NSUInteger)i renderCommandEncoder:(id<MTLRenderCommandEncoder>)rce
{
    __unsafe_unretained id<MTLTexture> textures[kMaxShaderBindings] = {NULL};
    id<MTLSamplerState>           samplers[kMaxShaderBindings]      = {NULL};
    for (ShaderPassTextureBinding *bind in _pass[i].bindings.textures) {
        NSUInteger binding = bind.binding;
        textures[binding] = *(bind.texture);
        samplers[binding] = _samplers[bind.filter][bind.wrap];
    }
    
    // enqueue commands
    [rce setRenderPipelineState:_pass[i].state];
    rce.label = _pass[i].state.label;

    const NSUInteger bOffsets[kMaxConstantBuffers] = { 0 };
    
    [rce setVertexBuffers:_pass[i].vBuffers offsets:bOffsets withRange:NSMakeRange(0, kMaxConstantBuffers)];
    [rce setFragmentBuffers:_pass[i].fBuffers offsets:bOffsets withRange:NSMakeRange(0, kMaxConstantBuffers)];
    [rce setFragmentTextures:textures withRange:NSMakeRange(0, kMaxShaderBindings)];
    [rce setFragmentSamplerStates:samplers withRange:NSMakeRange(0, kMaxShaderBindings)];
    [rce drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
}

- (void)renderFinalPassWithCommandEncoder:(id<MTLRenderCommandEncoder>)rce
{
    [rce setViewport:_outputFrame.viewport];
    if (_sourceTexture && _sourceTextureIsFlipped) {
        [rce setVertexBytes:_vertexFlipped length:sizeof(_vertexFlipped) atIndex:BufferIndexPositions];
    } else {
        [rce setVertexBytes:_vertex length:sizeof(_vertex) atIndex:BufferIndexPositions];
    }
    
    if (!_shader || _passCount == 0) {
        [self OE_renderTexture:_texture renderCommandEncoder:rce];
        return;
    }
    
    if (_pass[_lastPassIndex].renderTarget.view == nil) {
        // last pass renders directly to the final render target
        [self OE_renderPassIndex:_lastPassIndex renderCommandEncoder:rce];
    } else {
        [self OE_renderTexture:_pass[_lastPassIndex].renderTarget.view renderCommandEncoder:rce];
    }
}
- (void)OE_logResolvedDimensionsIfNeeded
{
    if (!_diagnosticPending || getenv("OE_FILTER_CHAIN_DIAGNOSTICS") == NULL) {
        return;
    }
    _diagnosticPending = NO;

    os_log_info(OE_LOG_DEFAULT,
                "filter diagnostics: drawable %.0fx%.0f output viewport (%.0f, %.0f) %.0fx%.0f, passes %lu",
                _drawableSize.width,
                _drawableSize.height,
                _outputFrame.viewport.originX,
                _outputFrame.viewport.originY,
                _outputFrame.viewport.width,
                _outputFrame.viewport.height,
                (unsigned long)_passCount);

    for (NSUInteger i = 0; i < _passCount; i++) {
        ShaderPass *pass = _shader.passes[i];
        id<MTLTexture> target = _pass[i].renderTarget.view;
        os_log_info(OE_LOG_DEFAULT,
                    "filter diagnostics: pass %lu scaleX=%{public}s scaleY=%{public}s viewport (%.0f, %.0f) %.0fx%.0f target %.0fx%.0f%@",
                    (unsigned long)i,
                    OEFilterChainScaleName(pass.scaleX),
                    OEFilterChainScaleName(pass.scaleY),
                    _pass[i].viewport.originX,
                    _pass[i].viewport.originY,
                    _pass[i].viewport.width,
                    _pass[i].viewport.height,
                    _pass[i].renderTarget.viewSize.x,
                    _pass[i].renderTarget.viewSize.y,
                    target == nil ? @" (direct/final)" : @"");
    }
}


- (void)OE_resizeRenderTargets
{
    if (!_shader || !_renderTargetsNeedResize) return;
    
    // release existing targets
    for (int i = 0; i < _passCount; i++) {
        _pass[i].renderTarget.view   = nil;
        _pass[i].feedbackTarget.view = nil;
        memset(&_pass[i].renderTarget.viewSize, 0, sizeof(_pass[i].renderTarget.viewSize));
        memset(&_pass[i].feedbackTarget.viewSize, 0, sizeof(_pass[i].feedbackTarget.viewSize));
    }
    
    // width and height represent the size of the Source image to the current
    // pass
    NSInteger width = _sourceRect.size.width, height = _sourceRect.size.height;
    
    CGSize viewportSize = CGSizeMake(_outputFrame.viewport.width, _outputFrame.viewport.height);
    
    for (unsigned i = 0; i < _passCount; i++) {
        ShaderPass *pass = _shader.passes[i];
        
        if (pass.isScaled) {
            switch (pass.scaleX) {
                case OEShaderPassScaleSource:
                    width *= pass.scale.width;
                    break;
                
                case OEShaderPassScaleViewport:
                    width = (NSInteger)(viewportSize.width * pass.scale.width);
                    break;
                
                case OEShaderPassScaleAbsolute:
                    width = (NSInteger)pass.size.width;
                    break;
                
                default:
                    break;
            }
            
            if (!width)
                width = viewportSize.width;
            
            switch (pass.scaleY) {
                case OEShaderPassScaleSource:
                    height *= pass.scale.height;
                    break;
                
                case OEShaderPassScaleViewport:
                    height = (NSInteger)(viewportSize.height * pass.scale.height);
                    break;
                
                case OEShaderPassScaleAbsolute:
                    // BOXER-FORK: absolute-Y scale must read height, not width.
                    height = (NSInteger)pass.size.height;                    break;
                
                default:
                    break;
            }
            
            if (!height)
                height = viewportSize.height;
        } else if (i == _lastPassIndex) {
            width  = viewportSize.width;
            height = viewportSize.height;
        }
        
        os_log_debug(OE_LOG_DEFAULT, "pass %d, render target size %lu x %lu", i, width, height);
        
        MTLPixelFormat fmt = _pass[i].bindings.format;
        if ((i != _lastPassIndex) ||
                (width != viewportSize.width) || (height != viewportSize.height) ||
                fmt != MTLPixelFormatBGRA8Unorm) {
            _pass[i].viewport.width  = width;
            _pass[i].viewport.height = height;
            _pass[i].viewport.znear  = 0.0;
            _pass[i].viewport.zfar   = 1.0;
            
            MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                          width:width
                                                                                         height:height
                                                                                      mipmapped:NO];
            td.storageMode = MTLStorageModePrivate;
            td.usage       = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            [self OE_initTexture:&_pass[i].renderTarget withDescriptor:td];
            NSString *label = [NSString stringWithFormat:@"Pass %02d Output", i];
            _pass[i].renderTarget.view.label = label;
            if (pass.isFeedback) {
                [self OE_initTexture:&_pass[i].feedbackTarget withDescriptor:td];
                _pass[i].feedbackTarget.view.label = label;
            }
        } else {
            // last pass can render directly to the output render target
            _pass[i].renderTarget.viewSize.x = width;
            _pass[i].renderTarget.viewSize.y = height;
            _pass[i].renderTarget.viewSize.z = 1.0f / width;
            _pass[i].renderTarget.viewSize.w = 1.0f / height;
        }
    }
    
    _renderTargetsNeedResize = NO;
}

- (void)OE_freeShaderResources
{
    for (int i = 0; i < kMaxShaderPasses; i++) {
        _pass[i].renderTarget.view   = nil;
        _pass[i].feedbackTarget.view = nil;
        _pass[i].bindings            = nil;
        _pass[i].state               = nil;
        
        memset(&_pass[i].renderTarget.viewSize, 0, sizeof(_pass[i].renderTarget.viewSize));
        memset(&_pass[i].feedbackTarget.viewSize, 0, sizeof(_pass[i].feedbackTarget.viewSize));
        
        for (unsigned j = 0; j < kMaxConstantBuffers; j++) {
            _pass[i].buffers[j] = nil;
        }
    }
    
    for (int i = 0; i < kMaxTextures; i++) {
        _luts[i].view = nil;
    }
    
    for (int i = 0; i <= kMaxFrameHistory; i++) {
        _sourceTextures[i].view = nil;
        memset(&_sourceTextures[i].viewSize, 0, sizeof(_sourceTextures[i].viewSize));
    }
    
    _historyCount  = 0;
    _passCount     = 0;
    _lastPassIndex = 0;
    _lutCount      = 0;
}

- (BOOL)setShaderFromURL:(NSURL *)url error:(NSError **)error
{
    if (error != NULL) {
        *error = nil;
    }

    os_log_debug(OE_LOG_DEFAULT, "loading shader from '%{public}s'", url.fileSystemRepresentation);
    _diagnosticPending = YES;

    CFTimeInterval start = CACurrentMediaTime();

    NSError     *err = nil;
    SlangShader *ss = [[SlangShader alloc] initFromURL:url error:&err];
    if (err != nil) {
        os_log_error(OE_LOG_DEFAULT, "unable to load shader '%{public}s: %{public}@", url.fileSystemRepresentation, err.localizedDescription);
        if (error) {
            *error = err;
        }
        return NO;
    }
    
    if (ss == nil) {
        if (error != NULL) {
            *error = err ?: OEFilterChainShaderError(@"The shader preset could not be parsed.");
        }
        return NO;
    }

    // Parsing can fail for malformed presets. Keep the active shader alive
    // until the replacement has parsed successfully.
    [self OE_freeShaderResources];
    
    _passCount     = ss.passes.count;
    _lastPassIndex = _passCount - 1;
    _lutCount      = ss.luts.count;
    
    OEShaderPassCompiler *compiler = [[OEShaderPassCompiler alloc] initWithShaderModel:ss];
    
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    
    @try {
        texture_t *source = &_sourceTextures[0];
        for (unsigned i = 0; i < _passCount; source = &_pass[i++].renderTarget) {
            ShaderPass *pass = ss.passes[i];
            
            ShaderPassSemantics *sem = [ShaderPassSemantics new];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&_sourceTextures[0].view
                       size:&_sourceTextures[0].viewSize
                   semantic:OEShaderTextureSemanticOriginal];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&source->view
                       size:&source->viewSize
                   semantic:OEShaderTextureSemanticSource];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&_sourceTextures[0].view stride:sizeof(*_sourceTextures)
                       size:&_sourceTextures[0].viewSize stride:sizeof(*_sourceTextures)
                   semantic:OEShaderTextureSemanticOriginalHistory];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&_pass[0].renderTarget.view stride:sizeof(*_pass)
                       size:&_pass[0].renderTarget.viewSize stride:sizeof(*_pass)
                   semantic:OEShaderTextureSemanticPassOutput];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&_pass[0].feedbackTarget.view stride:sizeof(*_pass)
                       size:&_pass[0].feedbackTarget.viewSize stride:sizeof(*_pass)
                   semantic:OEShaderTextureSemanticPassFeedback];
            [sem addTexture:(id<MTLTexture> __unsafe_unretained *)(void *)&_luts[0].view stride:sizeof(*_luts)
                       size:&_luts[0].viewSize stride:sizeof(*_luts)
                   semantic:OEShaderTextureSemanticUser];
            
            simd_float4x4 *mvp = (i == _lastPassIndex) ? &_uniforms.projectionMatrix : &_uniformsNoRotate.projectionMatrix;
           
            [sem addUniformData:mvp semantic:OEShaderBufferSemanticMVP];
            [sem addUniformData:&_pass[i].renderTarget.viewSize semantic:OEShaderBufferSemanticOutput];
            [sem addUniformData:&_outputFrame.outputSize semantic:OEShaderBufferSemanticFinalViewportSize];
            [sem addUniformData:&_pass[i].frameCount semantic:OEShaderBufferSemanticFrameCount];
            [sem addUniformData:&_pass[i].frameDirection semantic:OEShaderBufferSemanticFrameDirection];
            
            NSString *vs_src = nil;
            NSString *fs_src = nil;
            _pass[i].bindings = [ShaderPassBindings new];
            if (![compiler buildPass:i
                        metalVersion:options.languageVersion
                       passSemantics:sem
                        passBindings:_pass[i].bindings
                              vertex:&vs_src
                            fragment:&fs_src
                               error:&err]) {
                if (error) {
                    *error = err ?: OEFilterChainShaderError([NSString stringWithFormat:@"Unable to compile shader pass %u.", i]);
                }
                return NO;
            }
            
            _pass[i].hasFeedback   = pass.isFeedback;
            _pass[i].frameCountMod = (uint32_t)pass.frameCountMod;

#ifdef DEBUG
            BOOL save_msl = NO;
#else
            BOOL save_msl = NO;
#endif
            // vertex descriptor
            @try {
                MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
                vd.attributes[0].offset      = offsetof(Vertex, position);
                vd.attributes[0].format      = MTLVertexFormatFloat4;
                vd.attributes[0].bufferIndex = BufferIndexPositions;
                vd.attributes[1].offset      = offsetof(Vertex, texCoord);
                vd.attributes[1].format      = MTLVertexFormatFloat2;
                vd.attributes[1].bufferIndex = BufferIndexPositions;
                vd.layouts[4].stride         = sizeof(Vertex);
                vd.layouts[4].stepFunction   = MTLVertexStepFunctionPerVertex;
                
                MTLRenderPipelineDescriptor *psd = [MTLRenderPipelineDescriptor new];
                if (pass.alias.length > 0)
                {
                    psd.label = [NSString stringWithFormat:@"pass %d (%@)", i, pass.alias];
                }
                else
                {
                    psd.label = [NSString stringWithFormat:@"pass %d", i];
                }
                
                
                MTLRenderPipelineColorAttachmentDescriptor *ca = psd.colorAttachments[0];
                
                ca.pixelFormat                 = _pass[i].bindings.format;
                ca.blendingEnabled             = NO;
                ca.sourceAlphaBlendFactor      = MTLBlendFactorSourceAlpha;
                ca.sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
                ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                ca.destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
                
                psd.sampleCount      = 1;
                psd.vertexDescriptor = vd;
                
                NSError        *err;
                id<MTLLibrary> lib   = [_device newLibraryWithSource:vs_src options:options error:&err];
                if (err != nil) {
                    if (lib == nil) {
                        save_msl = YES;
                        os_log_error(OE_LOG_DEFAULT, "unable to compile vertex shader: %{public}@", err.localizedDescription);
                        // BOXER-FORK: mirror os_log shader-failure diagnostics to NSLog for interactive debugging.
                        BXShaderDiagLog(@"unable to compile vertex shader: %@", err.localizedDescription);
                        if (error) {
                            *error = err ?: OEFilterChainShaderError([NSString stringWithFormat:@"Unable to compile the vertex shader for pass %d.", i]);
                        }
                        return NO;
                    }
#if DEBUG_SHADER
                    os_log_debug(OE_LOG_DEFAULT, @"warnings compiling vertex shader: %@", err.localizedDescription);
#endif
                }
                
                psd.vertexFunction = [lib newFunctionWithName:@"main0"];
                
                lib = [_device newLibraryWithSource:fs_src options:nil error:&err];
                if (err != nil) {
                    if (lib == nil) {
                        save_msl = YES;
                        os_log_error(OE_LOG_DEFAULT, "unable to compile fragment shader: %{public}@", err.localizedDescription);
                        // BOXER-FORK: mirror os_log shader-failure diagnostics to NSLog for interactive debugging.
                        BXShaderDiagLog(@"unable to compile fragment shader: %@", err.localizedDescription);
                        if (error) {
                            *error = err ?: OEFilterChainShaderError([NSString stringWithFormat:@"Unable to compile the fragment shader for pass %d.", i]);
                        }
                        return NO;
                    }
#if DEBUG_SHADER
                    os_log_debug(OE_LOG_DEFAULT, "warnings compiling fragment shader: %{public}@", err.localizedDescription);
#endif
                }
                psd.fragmentFunction = [lib newFunctionWithName:@"main0"];
                
                _pass[i].state = [_device newRenderPipelineStateWithDescriptor:psd error:&err];
                if (err != nil) {
                    save_msl = YES;
                    os_log_error(OE_LOG_DEFAULT, "error creating pipeline state for pass %d: %{public}@", i, err.localizedDescription);
                    // BOXER-FORK: mirror os_log shader-failure diagnostics to NSLog for interactive debugging.
                    BXShaderDiagLog(@"error creating pipeline state for pass %d: %@", i, err.localizedDescription);
                    if (error) {
                        *error = err ?: OEFilterChainShaderError([NSString stringWithFormat:@"Unable to create the render pipeline for pass %d.", i]);
                    }
                    return NO;
                }
                // BOXER-FORK: Metal can return nil state without populating err.
                if (err == nil && _pass[i].state == nil) {
                    BXShaderDiagLog(@"pipeline state nil for pass %d (err == nil)", i);
                    if (error) {
                        *error = OEFilterChainShaderError([NSString stringWithFormat:@"Unable to create the render pipeline for pass %d.", i]);
                    }
                    return NO;
                }
                
                for (unsigned j = 0; j < kMaxConstantBuffers; j++) {
                    ShaderPassBufferBinding *sem = _pass[i].bindings.buffers[j];
                    assert(sem.bindingVert < kMaxConstantBuffers);
                    
                    size_t size = sem.size;
                    if (size == 0) {
                        continue;
                    }
                    MTLResourceOptions opts = _deviceUnifiedMemory
                        ? MTLResourceStorageModeShared
                        : MTLResourceStorageModeManaged;
                    id<MTLBuffer> buf = [_device newBufferWithLength:size options:opts];
                    _pass[i].buffers[j] = buf;
                    
                    if (sem.stageUsage & OEStageUsageVertex) {
                        _pass[i].vBuffers[sem.bindingVert] = buf;
                    }
                    if (sem.stageUsage & OEStageUsageFragment) {
                        _pass[i].fBuffers[sem.bindingFrag] = buf;
                    }
                }
            }
            @finally {
                if (save_msl) {
                    NSString *basePath = [pass.url.absoluteString stringByDeletingPathExtension];
                    
                    os_log_debug(OE_LOG_DEFAULT, "saving metal shader files to %{public}@", basePath);
                    
                    NSError *err = nil;
                    [vs_src writeToFile:[basePath stringByAppendingPathExtension:@"vs.metal"]
                             atomically:NO
                               encoding:NSStringEncodingConversionAllowLossy
                                  error:&err];
                    if (err != nil) {
                        os_log_error(OE_LOG_DEFAULT, "unable to save vertex shader source for pass %d: %{public}@", i, err.localizedDescription);
                    }
                    
                    err = nil;
                    [fs_src writeToFile:[basePath stringByAppendingPathExtension:@"fs.metal"]
                             atomically:NO
                               encoding:NSStringEncodingConversionAllowLossy
                                  error:&err];
                    if (err != nil) {
                        os_log_error(OE_LOG_DEFAULT, "unable to save fragment shader source for pass %d: %{public}@", i, err.localizedDescription);

                    }
                }
            }
        }

        _historyCount = compiler.historyCount;
        _shader       = ss;
        ss      = nil;
        [self loadLuts];
    }
    @finally {
        if (ss) {
            [self OE_freeShaderResources];
        }
        
        CFTimeInterval end = CACurrentMediaTime() - start;
        os_log_debug(OE_LOG_DEFAULT, "Shader compilation completed in %{xcode:interval}f seconds", end);
    }
    
    _renderTargetsNeedResize = YES;
    _historyNeedsInit        = YES;
    
    
    if (getenv("OE_FILTER_CHAIN_DIAGNOSTICS") != NULL) {
        // BOXER-FORK: report shader framebuffer formats once per successful load.
        NSMutableString *formats = [NSMutableString stringWithFormat:@"formats: source=%@ | history=%@ | passes=%lu",
                                     BXPixelFormatName(_sourceTexture ? _sourceTexture.pixelFormat : MTLPixelFormatInvalid),
                                     BXPixelFormatName(MTLPixelFormatBGRA8Unorm),
                                     (unsigned long)_passCount];
        for (NSUInteger i = 0; i < _passCount; i++) {
            ShaderPass *pass = _shader.passes[i];
            [formats appendFormat:@" | p%lu=%@%@", (unsigned long)i,
             BXPixelFormatName(_pass[i].bindings.format),
             pass.isFeedback ? @"(feedback)" : @""];
        }
        [formats appendFormat:@" | final=%@", BXPixelFormatName(_pass[_lastPassIndex].bindings.format)];
        BXShaderDiagLog(@"%@", formats);
    }
    
    return YES;
}

- (void)loadLuts {
    BOOL flipped = _sourceTexture && _sourceTextureIsFlipped;

    NSDictionary<MTKTextureLoaderOption, id> *opts = @{
                                                       MTKTextureLoaderOptionGenerateMipmaps: @YES,
                                                       MTKTextureLoaderOptionAllocateMipmaps: @YES,
                                                       MTKTextureLoaderOptionSRGB: @NO,
                                                       MTKTextureLoaderOriginFlippedVertically: @(flipped),
                                                       MTKTextureLoaderOptionTextureStorageMode: @(MTLStorageModePrivate),
                                                       };
    
    for (unsigned i = 0; i < _lutCount; i++) {
        ShaderLUT *lut   = _shader.luts[i];
        
        NSError        *err;
        id<MTLTexture> t = [_loader newTextureWithContentsOfURL:lut.url options:opts error:&err];
        if (err != nil) {
            // load a default texture so the failure is visibly obvious
            os_log_error(OE_LOG_DEFAULT, "unable to load LUT texture, using default. path '%{public}@: %{public}@", lut.url, err.localizedDescription);
            
            if (_checkers == nil) {
                /* Create a dummy texture instead. */
                const uint32_t T0 = 0xff000000u;
                const uint32_t T1 = 0xffffffffu;
                static const uint32_t checkerboard[] = {
                    T0, T1, T0, T1, T0, T1, T0, T1,
                    T1, T0, T1, T0, T1, T0, T1, T0,
                    T0, T1, T0, T1, T0, T1, T0, T1,
                    T1, T0, T1, T0, T1, T0, T1, T0,
                    T0, T1, T0, T1, T0, T1, T0, T1,
                    T1, T0, T1, T0, T1, T0, T1, T0,
                    T0, T1, T0, T1, T0, T1, T0, T1,
                    T1, T0, T1, T0, T1, T0, T1, T0,
                };
                
                CGContextRef ctx = CGBitmapContextCreate((void *)checkerboard, 8, 8, 8, 32, NSColorSpace.deviceRGBColorSpace.CGColorSpace, kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedLast);
                CGImageRef img = CGBitmapContextCreateImage(ctx);
                
                _checkers = [_loader newTextureWithCGImage:img options:opts error:&err];
                CGImageRelease(img);
                CGContextRelease(ctx);
            }
            
            t = _checkers;
        }
        
        [self OE_initTexture:&_luts[i] withTexture:t];
    }
}

@end
