import json
import sys
import os
import datetime
import warnings

import delightfulsoup as ds
import djangofy as dfy

import common

# -------------------------------------------------------------------------------


# Download image(s) from online source and translate 'src'
def wget_images(body, nb, name,
                dir_nb, img_folder_on_streambed, img_folder_in_repo):
    imgs = body.findAll('img')

    def custom_img_name(img_src, img_i):
        _, _ext = os.path.splitext(img_src)
        ext = _ext.split('?')[0]
        _img_i = str(img_i) if img_i >= 10 else '0' + str(img_i)
        return '{nb}_image{img_i}{ext}'.format(nb=nb, img_i=_img_i, ext=ext)

    def img_alt(img_src, img_i):
        _img_i = str(img_i) if img_i >= 10 else '0' + str(img_i)
        return "{name} image{img_i}".format(name=name, img_i=_img_i)

    ds.utils.wget_images(imgs,
                         dir_root=dir_nb,
                         dir_download=img_folder_in_repo,
                         dir_publish=img_folder_on_streambed,
                         translate_src=True,
                         custom_img_name=custom_img_name,
                         img_alt=img_alt)


# Remove h1 elements
def remove_h1(body):
    try:
        H1 = body.findAll('h1')

        H1[0].extract()  # Remove title

        for h1 in H1:  # swap h1 for h2
            h1.name = 'h2'

    except:
        warnings.warn("No H1 found")


# Add lightbox anchors around <img>
def add_lightbox(body):
    imgs = body.findAll('img')

    def a_href(img):
        return img['src']

    def a_data(img):
        return os.path.splitext(os.path.basename(img['src']))[0]

    ds.insert_around_nodes(imgs, 'a',
                           tag_attrs={'href': a_href,
                                      'data-lightbox': a_data})


# Translate root url + target blank on outbound href
def update_anchors(body):
    site_root = 'https://plot.ly'
    user_root = 'https://plot.ly/~'
    anchors = body.findAll('a')

    for anchor in anchors:
        if anchor.has_attr('href'):
            if anchor['href'].startswith((user_root, '#')):
                continue
            elif anchor['href'].startswith(site_root):
                ds.translate([anchor], 'href', {site_root: '/'})
                ds.translate([anchor], 'href', {'//': '/'})
            else:
                # Add target='_blank' attributes to outbound links
                ds.add_attr(anchor, {'target': '_blank'})


# Add anchors inside In / Out <div>
def add_in_out_anchors(body):
    divs = body.findAll('div', {'class': 'prompt'})

    def div_id(div):
        text = div.getText(strip=True, separator=u' ')
        if text:
            return text[:-1].replace(' ', '-').replace(u"\xa0", "-")

    def a_href(div):
        _id = div_id(div)
        if _id:
            return "#" + _id

    def a_class(div):
        return div['class']

    def a_content(div):
        return div.getText(strip=True, separator=u' ')

    ds.insert_inside_nodes(divs, 'a',
                           node_attrs={'id': div_id},
                           tag_attrs={'href': a_href,
                                      'class': a_class},
                           tag_content=a_content)


# -------------------------------------------------------------------------------


def append_config(config, arg, path_handler):

    now = datetime.datetime.now().strftime("%A %d %B %Y")
    NOTEBOOKS = path_handler.GLOBALS['NOTEBOOKS']
    github_page = path_handler.GLOBALS['GITHUB']['page'] + NOTEBOOKS
    github_raw = path_handler.GLOBALS['GITHUB']['raw'] + NOTEBOOKS

    config = dict(
        last_modified=now,
        title=config['title'],
        title_short=config['title_short'],
        meta_description=config['meta_description'],
        github_url=github_page + '/' + arg,
        file_ipynb=github_raw + '/' + arg + '/' + arg + '.ipynb',
        file_py=github_raw + '/' + arg + '/' + arg + '.py',
    )

    return config


def append_references(references, path_handler):

    includes = path_handler.GLOBALS['STREAMBED']['includes']

    references = dict(
        notebooks=references['notebooks'],
        paths=dict(),
        splash=[],
    )

    for nb in references['notebooks']:
        references['paths'][nb] = dict(
            body=os.path.join(includes, nb, 'body.html'),
            config=os.path.join(includes, nb, 'config.json'),
        )

        config = path_handler.load_config(nb)
        references['splash'] += [dict(
            title=config['title'],
            title_short=config['title_short'],
            relative_url=config['relative_url'],
            thumbnail_image=config['thumbnail_image']
        )]

    return references


# -------------------------------------------------------------------------------


def main():

    path_handler = common.PathHandler('publish')

    path_image = path_handler.GLOBALS['STREAMBED']['image']

    for arg in path_handler.args:

        dir_nb = path_handler.get_path(arg)
        file_html = path_handler.get_file(arg, '.tmp.html')
        config = path_handler.load_config(arg)
        tree = path_handler.get_tree(arg)

        body = ds.load_soup(file_html).body

        # Download images
        wget_images(body, arg, config['title_short'],
                    dir_nb, path_image, tree['static']['image'])

        # Update body
        remove_h1(body)
        update_anchors(body)
        add_lightbox(body)
        add_in_out_anchors(body)

        # Dump body
        ds.dump_soup(body, tree['includes']['nb']['body'], remove_tag='body')

        # Append and dump config
        config = append_config(config, arg, path_handler)
        ds.dump_json(config, tree['includes']['nb']['config'])

    # Append and dump references
    references = path_handler.load_references()
    references = append_references(references, path_handler)
    ds.dump_json(references, tree['includes']['references'])

    # Make url, sitemaps and redirect files
    nbs = path_handler.nbs
    relative_urls = path_handler.get_relative_urls()
    dfy.make_urls(nbs, relative_urls, tree['urls'],
                  app_name='api_docs',
                  class_name='IPythonNotebookPage')
    dfy.make_sitemaps(nbs, relative_urls, tree['sitemaps'])


if __name__ == "__main__":
    main()
