/**
 * @class ZipTileSource
 * @memberof OpenSeadragon
 * @extends OpenSeadragon.TileSource
 * @classdesc A tile source that loads tiles from a ZIP file.
 */
OpenSeadragon.ZipTileSource = function(options) {
    OpenSeadragon.TileSource.apply(this, [options]);
    
    this.zipFile = null;
    this.tileCache = {};
    this.loading = false;
};

OpenSeadragon.extend(OpenSeadragon.ZipTileSource.prototype, OpenSeadragon.TileSource.prototype, {
    /**
     * Initialize the tile source with a ZIP file
     * @param {File|Blob} zipFile - The ZIP file containing the tiles
     * @returns {Promise} A promise that resolves when the ZIP is loaded
     */
    initialize: function(zipFile) {
        var _this = this;
        this.zipFile = zipFile;
        
        return new Promise(function(resolve, reject) {
            JSZip.loadAsync(zipFile)
                .then(function(zip) {
                    _this.zip = zip;
                    resolve();
                })
                .catch(function(error) {
                    reject(error);
                });
        });
    },

    /**
     * Get the URL for a specific tile
     * @param {Number} level - The zoom level
     * @param {Number} x - The x coordinate
     * @param {Number} y - The y coordinate
     * @returns {String} The path to the tile in the ZIP file
     */
    getTileUrl: function(level, x, y) {
        return `helloworld_files/${level}/${x}_${y}.png`;
    },

    /**
     * Load a tile from the ZIP file
     * @param {Number} level - The zoom level
     * @param {Number} x - The x coordinate
     * @param {Number} y - The y coordinate
     * @returns {Promise} A promise that resolves with the tile data
     */
    loadTile: function(level, x, y) {
        var _this = this;
        var tilePath = this.getTileUrl(level, x, y);
        var cacheKey = `${level}_${x}_${y}`;
        
        // Check if tile is already in cache
        if (this.tileCache[cacheKey]) {
            return Promise.resolve(this.tileCache[cacheKey]);
        }
        
        return new Promise(function(resolve, reject) {
            _this.zip.file(tilePath).async('blob')
                .then(function(blob) {
                    var url = URL.createObjectURL(blob);
                    console.log("url", url);
                    _this.tileCache[cacheKey] = url;
                    resolve(url);
                })
                .catch(function(error) {
                    reject(error);
                });
        });
    },

    /**
     * Clean up resources
     */
    destroy: function() {
        // Revoke all object URLs
        for (var key in this.tileCache) {
            URL.revokeObjectURL(this.tileCache[key]);
        }
        this.tileCache = {};
        this.zip = null;
    }
}); 